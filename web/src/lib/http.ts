import { NextResponse } from "next/server";

import type { SourceContext } from "@/src/bff/types/baseContracts";

const DEFAULT_SOURCE_CONTEXT: SourceContext = {
  source_id: "spec_nyc_local_artifacts",
  source_type: "other"
};

type MetaOverrides = {
  source_context?: SourceContext;
  generated_at?: string;
};

export function okJson<T extends Record<string, unknown>>(
  data: T,
  status = 200,
  metaOverrides: MetaOverrides = {}
) {
  return NextResponse.json(
    {
      ...data,
      contract_version: "v1",
      generated_at: metaOverrides.generated_at ?? new Date().toISOString(),
      source_context: metaOverrides.source_context ?? DEFAULT_SOURCE_CONTEXT,
      request_id: crypto.randomUUID()
    },
    {
      status,
      headers: {
        "x-contract-version": "v1"
      }
    }
  );
}

export function errorJson(message: string, status = 400) {
  return NextResponse.json(
    {
      error: message,
      contract_version: "v1",
      generated_at: new Date().toISOString(),
      source_context: DEFAULT_SOURCE_CONTEXT,
      request_id: crypto.randomUUID()
    },
    {
      status,
      headers: {
        "x-contract-version": "v1"
      }
    }
  );
}
