import { expect, test } from "@playwright/test";

test("artifact explorer shows comparable-sales package evidence", async ({ page }) => {
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

  await page.goto("/artifacts");
  await expect(page.getByRole("heading", { name: /Reproducible model package evidence/i })).toBeVisible();
  await expect(page.getByText("Comparable Sales Evidence")).toBeVisible();
  await expect(page.getByText("comps_manifest.json")).toBeVisible();
  await expect(page.getByLabel("Selected comparable sales sample")).toBeVisible();

  expect(failedResponses).toEqual([]);
  expect(consoleErrors).toEqual([]);
});
