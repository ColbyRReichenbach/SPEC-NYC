import { mkdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";

import { expect, test, type Page } from "@playwright/test";

test("EDA lab renders governed artifacts and locks a hypothesis spec", async ({ page }, testInfo) => {
  const projectSlug = testInfo.project.name.replace(/[^a-z0-9]+/gi, "_").toLowerCase();
  const notebookPath = path.resolve(process.cwd(), "..", "reports", "eda", `playwright_eda_notebook_${projectSlug}.ipynb`);
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

  await mkdir(path.dirname(notebookPath), { recursive: true });
  await writeFile(
    notebookPath,
    JSON.stringify(
      {
        cells: [
          {
            cell_type: "markdown",
            source: ["# Playwright EDA Notebook\n", "Notebook artifact rendered in app."]
          },
          {
            cell_type: "code",
            execution_count: 1,
            source: ["print('eda')"],
            outputs: [{ name: "stdout", text: ["eda\n"] }]
          }
        ],
        metadata: {},
        nbformat: 4,
        nbformat_minor: 5
      },
      null,
      2
    )
  );

  try {
    await page.goto("/eda");
    await expect(page.getByRole("heading", { name: /Senior DS review surface/i })).toBeVisible();
    await expect(page.getByText("Latest EDA Run")).toBeVisible();
    await expect(page.getByRole("heading", { level: 2, name: "Underperforming Slices" })).toBeVisible();
    await expect(page.getByRole("heading", { level: 2, name: "Feature Effects By Slice" })).toBeVisible();
    await expect(page.locator("code").filter({ hasText: "reports/eda/avm_eda_report" }).first()).toBeVisible();
    await expectNoViewportOverflow(page);

    await expect(page.getByText("Artifact Index")).toBeVisible();
    await page.getByRole("link", { name: new RegExp(`playwright eda notebook ${projectSlug}`, "i") }).click();
    await expect(page.getByText("Artifact Viewer").first()).toBeVisible();
    await expect(page.getByText("Notebook artifact rendered in app.")).toBeVisible();
    await expect(page.getByText("eda", { exact: true }).first()).toBeVisible();
    await expectNoViewportOverflow(page);
    await page.goto("/eda");

    if (testInfo.project.name === "desktop") {
      await page.getByLabel("Collapse sidebar").click({ force: true });
      await expect(page.locator(".platform-shell")).toHaveClass(/is-collapsed/);
      await expectNoViewportOverflow(page);
    }

    await page.getByRole("button", { name: /Lock Hypothesis Spec/i }).first().click({ force: true });
    await expect(page.getByText("Locked spec written to experiment registry.")).toBeVisible();
    await expect(page.getByRole("link", { name: /Open run/i })).toBeVisible();
    await expectNoViewportOverflow(page);

    expect(failedResponses).toEqual([]);
    expect(consoleErrors).toEqual([]);
  } finally {
    await rm(notebookPath, { force: true });
  }
});

async function expectNoViewportOverflow(page: Page) {
  const result = await page.evaluate(() => {
    const root = document.documentElement;
    const offenders = Array.from(document.querySelectorAll<HTMLElement>("body *"))
      .filter((element) => {
        const style = window.getComputedStyle(element);
        if (style.position === "fixed" || style.overflowX === "auto" || style.overflowX === "scroll") {
          return false;
        }
        const rect = element.getBoundingClientRect();
        return rect.right > window.innerWidth + 1 || rect.left < -1;
      })
      .slice(0, 5)
      .map((element) => ({
        tag: element.tagName.toLowerCase(),
        className: String(element.className),
        text: element.textContent?.trim().slice(0, 80) ?? ""
      }));

    return {
      scrollWidth: root.scrollWidth,
      clientWidth: root.clientWidth,
      offenders
    };
  });

  expect(result.scrollWidth, JSON.stringify(result.offenders, null, 2)).toBeLessThanOrEqual(result.clientWidth + 1);
}
