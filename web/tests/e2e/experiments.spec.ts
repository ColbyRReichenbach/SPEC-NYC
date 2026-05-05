import { expect, test } from "@playwright/test";

test("experiment workbench creates a locked auditable preflight run", async ({ page }, testInfo) => {
  const failedResponses: string[] = [];
  const consoleErrors: string[] = [];
  page.on("response", (response) => {
    if (response.status() >= 400) {
      failedResponses.push(`${response.status()} ${response.url()}`);
    }
  });
  page.on("console", (message) => {
    if (message.type() === "error") {
      consoleErrors.push(message.text());
    }
  });

  await page.goto("/experiments");
  await expect(page.getByRole("heading", { name: /Hypothesis-led model development/i })).toBeVisible();
  await expect(page.getByLabel("Locked spec controls")).toContainText("same rows");
  await expect(page.getByText("Previous Runs")).toBeVisible();

  if (testInfo.project.name === "desktop") {
    await page.getByLabel("Collapse sidebar").click({ force: true });
    await expect(page.locator(".platform-shell")).toHaveClass(/is-collapsed/);
  }

  const hypothesis = `Playwright locked spec should create auditable artifacts ${Date.now()}.`;
  await page.getByLabel("Hypothesis").fill(hypothesis);
  await page.getByRole("button", { name: /Run Experiment Preflight/i }).click({ force: true });

  await expect(page.locator(".manifest-preview")).toBeVisible();
  await expect(page.locator(".manifest-card code")).toContainText("run_manifest.json");
  await expect(page.locator(".manifest-preview")).toContainText("comparison_report.json");
  await expect(page.locator(".manifest-preview")).toContainText("split_signature_sha256");
  await expect(page.locator(".manifest-preview")).toContainText("same_dataset_required");
  await expect(page.locator(".experiment-run-card").filter({ hasText: hypothesis })).toContainText(/spec locked/i);

  await page.getByRole("button", { name: /Queue for Review/i }).click({ force: true });
  await expect(page.locator(".manifest-preview")).toContainText("review_requested");
  await expect(page.getByText("Review request written.")).toBeVisible();

  await page.getByRole("button", { name: /Approve Review/i }).click({ force: true });
  await expect(page.locator(".manifest-preview")).toContainText("review_approved");
  await expect(page.getByText("Review approval written.")).toBeVisible();

  await page.getByRole("button", { name: /Queue Training/i }).click({ force: true });
  await expect(page.locator(".manifest-preview")).toContainText("training_job");
  await expect(page.locator(".manifest-preview")).toContainText("python3 -m src.model");
  await expect(page.locator(".manifest-preview")).toContainText("segmented_router");
  await expect(page.getByLabel("Governed experiment queues")).toContainText("Experiment Queue");
  await expect(page.getByLabel("Governed experiment queues")).toContainText("queued");

  expect(failedResponses).toEqual([]);
  expect(consoleErrors).toEqual([]);
});
