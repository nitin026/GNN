import { expect, test } from "@playwright/test";

test("load the page, move the slider, see an alert", async ({ page }) => {
  await page.goto("/?case=amphan_replay&view=alerts");
  await expect(page.locator(".badge.syn")).toHaveText("SYNTHETIC");
  const slider = page.getByLabel("Lead time (hours)");
  await slider.fill("96");
  await expect(page.getByText("Alerts (+96 h)", { exact: false })).toBeVisible();
  const firstRow = page.locator("table").first().locator("tbody tr").first();
  await expect(firstRow).toBeVisible();
  await firstRow.click();
  await expect(page.locator(".explain")).toContainText("within 50 km");
  await expect(page.locator(".map canvas").first()).toBeVisible();
});

test("all four views render", async ({ page }) => {
  await page.goto("/?case=cyc_04&lead=48");
  for (const v of ["Operations", "Downscaling", "Alerts", "Model performance"]) {
    await page.getByRole("tab", { name: v }).click();
    await expect(page.locator(`section[aria-label="${v} view"]`)).toBeVisible();
  }
  await expect(page.getByText("Amphan track error vs IBTrACS")).toBeVisible();
});
