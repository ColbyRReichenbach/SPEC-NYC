import { monitoringOverviewSchema } from "@/src/features/monitoring/schemas/monitoringSchemas";

export async function fetchMonitoringOverview(window = "30d") {
  const response = await fetch(`/api/v1/monitoring/overview?window=${encodeURIComponent(window)}`, {
    cache: "no-store"
  });

  if (!response.ok) {
    throw new Error(`Monitoring overview request failed with status ${response.status}`);
  }

  const data = await response.json();
  return monitoringOverviewSchema.parse(data);
}
