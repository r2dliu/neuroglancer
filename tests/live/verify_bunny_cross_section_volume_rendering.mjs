/* global globalThis */

/**
 * Opt-in live smoke test for cross-section slab rendering against the public
 * OME-Zarr 0.5 bunny dataset.
 *
 * Start a development server first, then run:
 *
 *   node tests/live/verify_bunny_cross_section_volume_rendering.mjs
 *
 * Set NEUROGLANCER_LIVE_VIEWER_URL to test a server other than
 * http://127.0.0.1:8080/.
 */

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { env, stdout } from "node:process";
import { URL } from "node:url";
import { chromium } from "playwright";

const viewerUrl = env.NEUROGLANCER_LIVE_VIEWER_URL ?? "http://127.0.0.1:8080/";
const datasetUrl =
  "https://ome-zarr-scivis.s3.us-east-1.amazonaws.com/v0.5/96x2/bunny.ome.zarr";
const timeout = Number(env.NEUROGLANCER_LIVE_TIMEOUT_MS ?? 120_000);
const slabVoxelRange = 10;

function makeState(voxelRange, volumeRenderingMode) {
  return {
    position: [0, 0, 0],
    crossSectionScale: 0.5,
    crossSectionVoxelRange: voxelRange,
    crossSectionVolumeRenderingMode: volumeRenderingMode,
    layers: [
      {
        type: "image",
        source: datasetUrl,
        shader:
          "#uicontrol invlerp normalized(range=[0, 63536])\n" +
          "void main() { emitGrayscale(normalized()); }",
        name: "bunny",
      },
    ],
    showAxisLines: false,
    showScaleBar: false,
    prefetch: false,
    layout: "xy",
  };
}

async function waitForViewerReady(page) {
  const options = { polling: 100, timeout };
  await page.waitForFunction(
    () => {
      const viewer = globalThis.viewer;
      if (viewer === undefined) return false;
      const layers = viewer.layerManager.managedLayers;
      return (
        layers.length === 1 &&
        layers[0].layer !== null &&
        layers[0].isReady() &&
        viewer.isReady()
      );
    },
    undefined,
    options,
  );

  // Match the screenshot handler's readiness rule: the viewer must remain
  // ready after pending rendering and chunk-priority work has run.
  await page.evaluate(
    () =>
      new Promise((resolve) =>
        globalThis.requestAnimationFrame(() =>
          globalThis.requestAnimationFrame(resolve),
        ),
      ),
  );
  await page.waitForFunction(
    () => globalThis.viewer !== undefined && globalThis.viewer.isReady(),
    undefined,
    options,
  );
}

function getChunkKey(url) {
  const match = new URL(url).pathname.match(
    /\/(scale[0-9]+\/bunny\/c\/[^?#]+)$/,
  );
  return match?.[1];
}

function getZChunkIndices(chunkKeys) {
  const result = {};
  for (const key of chunkKeys) {
    const [scale, , , ...chunkCoordinates] = key.split("/");
    // Neuroglancer's Zarr chunk keys are stored in the reverse of the OME
    // dimension order, making z the last coordinate for this dataset.
    const z = chunkCoordinates.at(-1);
    (result[scale] ??= new Set()).add(Number(z));
  }
  return Object.fromEntries(
    Object.entries(result).map(([scale, indices]) => [
      scale,
      [...indices].sort((a, b) => a - b),
    ]),
  );
}

async function render(browser, voxelRange, volumeRenderingMode) {
  const page = await browser.newPage({
    viewport: { width: 512, height: 512 },
  });
  const successfulUrls = new Set();
  const chunkKeys = new Set();
  const pageErrors = [];

  page.on("pageerror", (error) => pageErrors.push(String(error)));
  page.on("response", (response) => {
    const url = response.url();
    if (!url.startsWith(datasetUrl) || response.status() !== 200) return;
    successfulUrls.add(url);
    const chunkKey = getChunkKey(url);
    if (chunkKey !== undefined) chunkKeys.add(chunkKey);
  });

  const state = makeState(voxelRange, volumeRenderingMode);
  const url = `${viewerUrl.replace(/\/$/, "")}/#!${encodeURIComponent(
    JSON.stringify(state),
  )}`;
  try {
    await page.goto(url, { waitUntil: "domcontentloaded", timeout });
    await waitForViewerReady(page);

    const runtimeState = await page.evaluate(() => ({
      voxelRange: globalThis.viewer.crossSectionVoxelRange.value,
      volumeRenderingMode:
        globalThis.viewer.crossSectionVolumeRenderingMode.toJSON() ?? "max",
    }));
    assert.deepEqual(runtimeState, { voxelRange, volumeRenderingMode });

    const requiredMetadataUrls = [
      `${datasetUrl}/zarr.json`,
      ...[0, 1, 2].map(
        (scale) => `${datasetUrl}/scale${scale}/bunny/zarr.json`,
      ),
    ];
    for (const requiredUrl of requiredMetadataUrls) {
      assert.ok(
        successfulUrls.has(requiredUrl),
        `Did not load required metadata: ${requiredUrl}`,
      );
    }
    assert.ok(chunkKeys.size > 0, "No bunny data chunks were downloaded");
    assert.deepEqual(pageErrors, []);

    const screenshot = await page.screenshot();
    return {
      chunkKeys,
      screenshot,
      screenshotSha256: createHash("sha256").update(screenshot).digest("hex"),
      zChunkIndices: getZChunkIndices(chunkKeys),
    };
  } finally {
    await page.close();
  }
}

const browser = await chromium.launch({ headless: true });
try {
  const centerPlane = await render(browser, 0, "max");
  const maximumSlab = await render(browser, slabVoxelRange, "max");
  const minimumSlab = await render(browser, slabVoxelRange, "min");

  assert.ok(
    maximumSlab.chunkKeys.size > centerPlane.chunkKeys.size,
    `Expected voxel range ${slabVoxelRange} to download more chunks than ` +
      `range 0, but observed ${maximumSlab.chunkKeys.size} and ` +
      `${centerPlane.chunkKeys.size}`,
  );
  assert.notDeepEqual(
    maximumSlab.screenshot,
    minimumSlab.screenshot,
    "Min and max slab rendering produced identical screenshots",
  );

  stdout.write(
    JSON.stringify(
      {
        status: "passed",
        viewerUrl,
        datasetUrl,
        centerPlaneChunks: centerPlane.chunkKeys.size,
        slabChunks: maximumSlab.chunkKeys.size,
        centerPlaneZChunkIndices: centerPlane.zChunkIndices,
        slabZChunkIndices: maximumSlab.zChunkIndices,
        maximumScreenshotSha256: maximumSlab.screenshotSha256,
        minimumScreenshotSha256: minimumSlab.screenshotSha256,
      },
      undefined,
      2,
    ) + "\n",
  );
} finally {
  await browser.close();
}
