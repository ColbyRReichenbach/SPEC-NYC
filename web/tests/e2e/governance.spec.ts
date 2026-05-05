import { expect, test } from "@playwright/test";

test("governance workbench creates and rejects a release proposal", async ({ page }) => {
  const initial = await page.request.get("/api/v1/governance/proposals");
  const initialState = (await initial.json()) as { eligibleExperiments?: unknown[] };
  test.skip((initialState.eligibleExperiments ?? []).length === 0, "No completed passed experiment exists for proposal QA.");

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

  await page.goto("/governance");
  await expect(page.getByRole("heading", { name: /Release proposal workflow/i })).toBeVisible();
  await expect(page.getByLabel("Eligible completed experiment")).toBeVisible();

  const createProposalButton = page.getByRole("button", { name: /Create Release Proposal/i });
  await expect(createProposalButton).toBeEnabled();
  await createProposalButton.click();
  await expect(page.getByText("Release proposal created.")).toBeVisible();

  await page.getByLabel("Decision reason").fill("Rejected by Playwright governance QA.");
  const rejectButton = page.getByRole("button", { name: /^Reject$/i });
  await expect(rejectButton).toBeEnabled();
  await rejectButton.click();
  await expect(page.getByText("Release proposal rejected.")).toBeVisible();
  await expect(page.getByLabel("Release proposals")).toContainText("rejected");

  expect(failedResponses).toEqual([]);
  expect(consoleErrors).toEqual([]);
});
