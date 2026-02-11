"""Model training pipeline (W2 baseline) for S.P.E.C. NYC."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from config.settings import MODEL_CONFIG, MODELS_DIR
from src.database import engine
from src.evaluate import build_segment_scorecard, evaluate_predictions, mdape, save_metrics
from src.explain import generate_shap_artifacts
from src.mlops.track_run import get_git_sha, log_run


logger = logging.getLogger(__name__)

TARGET_COL = "sale_price"
DATE_COL = "sale_date"

NUMERIC_FEATURES = [
    "gross_square_feet",
    "year_built",
    "building_age",
    "residential_units",
    "total_units",
    "distance_to_center_km",
    "h3_price_lag",
]

CATEGORICAL_FEATURES = [
    "borough",
    "building_class",
    "property_segment",
    "price_tier",
    "neighborhood",
]

BASE_REQUIRED_COLUMNS = [
    DATE_COL,
    TARGET_COL,
    "h3_index",
    "gross_square_feet",
    "year_built",
    "building_age",
    "residential_units",
    "total_units",
    "distance_to_center_km",
    "borough",
    "building_class",
    "property_segment",
    "price_tier",
    "neighborhood",
]


@dataclass
class TrainConfig:
    model_version: str = "v1"
    test_size: float = 0.2
    random_state: int = 42
    optuna_trials: int = 0
    limit: int | None = None
    generate_shap: bool = True
    shap_sample_size: int = 500
    enable_mlflow: bool = True
    dataset_version: str | None = None
    tracking_uri: str | None = None


def load_training_data(input_csv: Path | None = None, limit: int | None = None) -> pd.DataFrame:
    """Load model training data from CSV or Postgres."""
    if input_csv is not None:
        if not input_csv.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")
        logger.info("Loading training data from CSV: %s", input_csv)
        df = pd.read_csv(input_csv, low_memory=False)
    else:
        logger.info("Loading training data from PostgreSQL 'sales' table")
        query = """
            SELECT
                sale_date,
                sale_price,
                h3_index,
                gross_square_feet,
                year_built,
                building_age,
                residential_units,
                total_units,
                distance_to_center_km,
                borough,
                building_class,
                property_segment,
                price_tier,
                neighborhood
            FROM sales
            WHERE sale_price IS NOT NULL
              AND sale_date IS NOT NULL
        """
        df = pd.read_sql(query, engine)

    if limit is not None:
        df = df.head(limit).copy()
        logger.info("Applied training row limit: %s", limit)
    logger.info("Loaded %s rows", len(df))
    return df


def prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize and validate the training frame."""
    frame = df.copy()
    frame.columns = [c.strip().lower() for c in frame.columns]

    missing = [c for c in BASE_REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Training data missing required columns: {missing}")

    frame[DATE_COL] = pd.to_datetime(frame[DATE_COL], errors="coerce")
    frame[TARGET_COL] = pd.to_numeric(frame[TARGET_COL], errors="coerce")
    frame = frame.dropna(subset=[DATE_COL, TARGET_COL, "h3_index"])
    frame = frame[frame[TARGET_COL] >= 10_000].copy()
    frame = frame.sort_values(DATE_COL).reset_index(drop=True)
    if len(frame) < 50:
        raise ValueError("Need at least 50 rows after filtering for stable baseline training.")
    return frame


def time_split(frame: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time order to prevent leakage."""
    if not 0 < test_size < 0.5:
        raise ValueError("test_size must be between 0 and 0.5")
    split_idx = int(len(frame) * (1 - test_size))
    train_df = frame.iloc[:split_idx].copy()
    test_df = frame.iloc[split_idx:].copy()
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Time split produced empty train/test dataframes.")
    return train_df, test_df


def add_h3_price_lag(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute h3 price lag from training data only and map into train/test.
    This is a leakage-safe proxy until neighborhood-neighbor aggregation is added.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    h3_median = train_df.groupby("h3_index")[TARGET_COL].median()
    global_median = float(train_df[TARGET_COL].median())

    train_df["h3_price_lag"] = train_df["h3_index"].map(h3_median).fillna(global_median)
    test_df["h3_price_lag"] = test_df["h3_index"].map(h3_median).fillna(global_median)
    return train_df, test_df


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore")
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_pipeline(params: Dict[str, object], random_state: int) -> Pipeline:
    """Construct model pipeline with preprocessing + XGBoost regressor."""
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    model = XGBRegressor(
        n_estimators=int(params.get("n_estimators", MODEL_CONFIG.n_estimators)),
        max_depth=int(params.get("max_depth", MODEL_CONFIG.max_depth)),
        learning_rate=float(params.get("learning_rate", MODEL_CONFIG.learning_rate)),
        min_child_weight=float(params.get("min_child_weight", MODEL_CONFIG.min_child_weight)),
        subsample=float(params.get("subsample", MODEL_CONFIG.subsample)),
        colsample_bytree=float(params.get("colsample_bytree", MODEL_CONFIG.colsample_bytree)),
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=random_state,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def _objective(
    trial: optuna.Trial,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    random_state: int,
) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }
    pipe = build_pipeline(params, random_state=random_state)

    x_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df[TARGET_COL]
    x_valid = valid_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_valid = valid_df[TARGET_COL]

    pipe.fit(x_train, y_train)
    preds = pipe.predict(x_valid)
    return mdape(y_valid, preds)


def tune_hyperparameters(train_df: pd.DataFrame, random_state: int, n_trials: int) -> Dict[str, object]:
    """Run Optuna tuning on a time-based train/validation split."""
    if n_trials <= 0:
        return {}

    split_idx = int(len(train_df) * 0.85)
    fit_df = train_df.iloc[:split_idx].copy()
    valid_df = train_df.iloc[split_idx:].copy()
    if len(valid_df) < 30:
        logger.warning("Validation slice too small for reliable tuning; skipping Optuna.")
        return {}

    logger.info("Starting Optuna tuning with %s trials", n_trials)
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: _objective(trial, fit_df, valid_df, random_state), n_trials=n_trials)
    logger.info("Optuna best MdAPE: %.4f", study.best_value)
    logger.info("Optuna best params: %s", study.best_params)
    return study.best_params


def train_model(df: pd.DataFrame, config: TrainConfig) -> dict:
    """Train baseline model, evaluate, and persist W2 artifacts."""
    frame = prepare_training_frame(df)
    train_df, test_df = time_split(frame, config.test_size)
    train_df, test_df = add_h3_price_lag(train_df, test_df)

    best_params = tune_hyperparameters(train_df, config.random_state, config.optuna_trials)
    pipeline = build_pipeline(best_params, random_state=config.random_state)

    x_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df[TARGET_COL]
    x_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    logger.info("Training baseline XGBoost model on %s rows", len(train_df))
    pipeline.fit(x_train, y_train)
    preds = pipeline.predict(x_test)

    eval_df = test_df.copy()
    eval_df["predicted_price"] = preds
    eval_df["prediction_error"] = eval_df["predicted_price"] - eval_df[TARGET_COL]
    eval_df["abs_pct_error"] = np.abs(eval_df["prediction_error"] / eval_df[TARGET_COL])

    metrics = evaluate_predictions(eval_df)
    segment_scores = [v["ppe10"] for v in metrics.get("per_segment", {}).values() if isinstance(v, dict) and "ppe10" in v]
    segment_ppe10_variance = float(max(segment_scores) - min(segment_scores)) if len(segment_scores) >= 2 else 0.0
    segment_variance_flag_v2 = bool(segment_ppe10_variance > 0.15)
    metrics["metadata"] = {
        "model_version": config.model_version,
        "trained_at_utc": datetime.utcnow().isoformat(),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_columns": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "optuna_trials": int(config.optuna_trials),
        "best_params": best_params,
        "segment_ppe10_variance": segment_ppe10_variance,
        "segment_variance_flag_v2": segment_variance_flag_v2,
    }
    if segment_variance_flag_v2:
        logger.warning("Segment PPE10 variance %.3f exceeds 0.15. Flagging V2 segment-specific models.", segment_ppe10_variance)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    report_dir = Path("reports/model")
    report_dir.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"model_{config.model_version}.joblib"
    metrics_path = MODELS_DIR / f"metrics_{config.model_version}.json"
    scorecard_path = report_dir / f"segment_scorecard_{config.model_version}.csv"
    predictions_path = report_dir / f"evaluation_predictions_{config.model_version}.csv"

    artifact = {
        "pipeline": pipeline,
        "feature_columns": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "model_version": config.model_version,
        "trained_at_utc": datetime.utcnow().isoformat(),
    }
    joblib.dump(artifact, model_path)
    save_metrics(metrics, metrics_path)
    build_segment_scorecard(eval_df).to_csv(scorecard_path, index=False)
    eval_df.to_csv(predictions_path, index=False)

    logger.info("Saved model artifact: %s", model_path)
    logger.info("Saved metrics artifact: %s", metrics_path)
    logger.info("Saved segment scorecard: %s", scorecard_path)
    logger.info("Saved evaluation predictions: %s", predictions_path)

    shap_artifacts = {}
    if config.generate_shap:
        shap_artifacts = generate_shap_artifacts(
            model_path=model_path,
            evaluation_csv=predictions_path,
            summary_plot_path=report_dir / f"shap_summary_{config.model_version}.png",
            waterfall_plot_path=report_dir / f"shap_waterfall_{config.model_version}.png",
            sample_size=config.shap_sample_size,
            random_state=config.random_state,
        )
        logger.info("Generated SHAP artifacts: %s", shap_artifacts)

    mlflow_result = {}
    if config.enable_mlflow:
        inferred_dataset_version = config.dataset_version or (
            f"rows_{len(frame)}_date_{frame[DATE_COL].min().date()}_{frame[DATE_COL].max().date()}"
        )
        try:
            mlflow_result = log_run(
                metrics_json=metrics_path,
                model_artifact=model_path,
                scorecard_csv=scorecard_path,
                predictions_csv=predictions_path,
                experiment_name="spec-nyc-avm",
                run_name=f"train-{config.model_version}",
                dataset_version=inferred_dataset_version,
                git_sha=get_git_sha(),
                tracking_uri=config.tracking_uri,
            )
            logger.info("Tracked MLflow run: %s", mlflow_result)
        except Exception as exc:
            logger.warning("MLflow tracking failed: %s", exc)

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "scorecard_path": str(scorecard_path),
        "predictions_path": str(predictions_path),
        "metrics": metrics,
        "shap_artifacts": shap_artifacts,
        "mlflow": mlflow_result,
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Train baseline NYC AVM model (W2).")
    parser.add_argument("--input-csv", type=Path, default=None, help="Optional CSV input; defaults to Postgres sales table")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for deterministic/dev runs")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--optuna-trials", type=int, default=0)
    parser.add_argument("--model-version", type=str, default="v1")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-shap", action="store_true", help="Disable SHAP artifact generation")
    parser.add_argument("--shap-sample-size", type=int, default=500)
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow run logging")
    parser.add_argument("--dataset-version", type=str, default=None)
    parser.add_argument("--tracking-uri", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    config = TrainConfig(
        model_version=args.model_version,
        test_size=args.test_size,
        random_state=args.random_state,
        optuna_trials=args.optuna_trials,
        limit=args.limit,
        generate_shap=not args.no_shap,
        shap_sample_size=args.shap_sample_size,
        enable_mlflow=not args.no_mlflow,
        dataset_version=args.dataset_version,
        tracking_uri=args.tracking_uri,
    )

    df = load_training_data(input_csv=args.input_csv, limit=args.limit)
    result = train_model(df, config)

    summary = {
        "artifacts": {
            "model": result["model_path"],
            "metrics": result["metrics_path"],
            "segment_scorecard": result["scorecard_path"],
            "predictions": result["predictions_path"],
        },
        "overall_metrics": result["metrics"]["overall"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _cli()
