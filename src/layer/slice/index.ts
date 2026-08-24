import type { ManagedUserLayer } from "#src/layer/index.js";
import { registerLayerType, UserLayer } from "#src/layer/index.js";
import type { SliceParameters } from "#src/slice_projection/base.js";
import { SliceProjectionMode } from "#src/slice_projection/base.js";
import { SliceProjectionRenderLayer } from "#src/slice_projection/frontend.js";
import { ImageRenderLayer } from "#src/sliceview/volume/image_renderlayer.js";
import type { WatchableValueInterface } from "#src/trackable_value.js";
import { WatchableValue } from "#src/trackable_value.js";
import { parseRGBColorSpecification, serializeColor } from "#src/util/color.js";
import type { Borrowed } from "#src/util/disposable.js";
import type { vec3 } from "#src/util/geom.js";
import { quat } from "#src/util/geom.js";
import {
  parseFixedLengthArray,
  verifyEnumString,
  verifyFiniteFloat,
  verifyFinitePositiveFloat,
  verifyNonnegativeInt,
  verifyObject,
  verifyOptionalObjectProperty,
} from "#src/util/json.js";
import { NullarySignal } from "#src/util/signal.js";
import type { Trackable } from "#src/util/trackable.js";

const SLICE_JSON_KEY = "slice";
const SOURCE_JSON_KEY = "source";

const DEFAULT_POSITION = [0, 0, 0];
const DEFAULT_ORIENTATION = [0, 0, 0, 1];
const DEFAULT_VOXEL_RANGE = 0;
const DEFAULT_PROJECTION_MODE = SliceProjectionMode.MAX;
const DEFAULT_BACKGROUND_COLOR = "#808080";
const DEFAULT_WIDTH = 1000;
const DEFAULT_HEIGHT = 1000;

function defaultSliceParameters(): SliceParameters {
  return {
    position: Float32Array.from(DEFAULT_POSITION),
    orientation: Float32Array.from(DEFAULT_ORIENTATION),
    voxelRange: DEFAULT_VOXEL_RANGE,
    projectionMode: DEFAULT_PROJECTION_MODE,
    backgroundColor: parseRGBColorSpecification(DEFAULT_BACKGROUND_COLOR),
    width: DEFAULT_WIDTH,
    height: DEFAULT_HEIGHT,
  };
}

export class TrackableSliceParameters
  implements Trackable, WatchableValueInterface<SliceParameters>
{
  changed = new NullarySignal();
  value = defaultSliceParameters();

  reset() {
    this.value = defaultSliceParameters();
    this.changed.dispatch();
  }

  restoreState(x: unknown) {
    const value = defaultSliceParameters();
    if (x !== undefined && x !== null) {
      verifyObject(x);
      verifyOptionalObjectProperty(x, "position", (position) =>
        parseFixedLengthArray(value.position, position, verifyFiniteFloat),
      );
      verifyOptionalObjectProperty(x, "orientation", (orientation) => {
        parseFixedLengthArray(
          value.orientation,
          orientation,
          verifyFiniteFloat,
        );
        quat.normalize(
          value.orientation as unknown as quat,
          value.orientation as unknown as quat,
        );
      });
      verifyOptionalObjectProperty(x, "voxelRange", (voxelRange) => {
        value.voxelRange = verifyNonnegativeInt(voxelRange);
      });
      verifyOptionalObjectProperty(x, "projectionMode", (projectionMode) => {
        value.projectionMode = verifyEnumString(
          projectionMode,
          SliceProjectionMode,
        );
      });
      verifyOptionalObjectProperty(x, "backgroundColor", (backgroundColor) => {
        value.backgroundColor = parseRGBColorSpecification(backgroundColor);
      });
      verifyOptionalObjectProperty(x, "size", (size) => {
        verifyObject(size);
        verifyOptionalObjectProperty(size, "width", (width) => {
          value.width = verifyFinitePositiveFloat(width);
        });
        verifyOptionalObjectProperty(size, "height", (height) => {
          value.height = verifyFinitePositiveFloat(height);
        });
      });
    }
    this.value = value;
    this.changed.dispatch();
  }

  toJSON() {
    const value = this.value;
    return {
      position: Array.from(value.position),
      orientation: Array.from(value.orientation),
      voxelRange: value.voxelRange,
      projectionMode: SliceProjectionMode[value.projectionMode].toLowerCase(),
      backgroundColor: serializeColor(value.backgroundColor as vec3),
      size: { width: value.width, height: value.height },
    };
  }
}

export class SliceUserLayer extends UserLayer {
  slice = new TrackableSliceParameters();
  imageSources = new WatchableValue<readonly ImageRenderLayer[]>([]);

  constructor(managedLayer: Borrowed<ManagedUserLayer>) {
    super(managedLayer);
    this.slice.changed.add(this.specificationChanged.dispatch);
    this.registerDisposer(
      this.manager.rootLayers.layersChanged.add(() =>
        this.updateImageSources(),
      ),
    );
    this.registerDisposer(() => {
      for (const source of this.imageSources.value) source.dispose();
    });
    this.addRenderLayer(
      new SliceProjectionRenderLayer({
        chunkManager: this.manager.chunkManager,
        sliceParameters: this.slice,
        imageSources: this.imageSources,
      }),
    );
    this.updateImageSources();
  }

  canAddDataSource() {
    return false;
  }

  getDataSourceSpecifications() {
    return [];
  }

  private updateImageSources() {
    const sources: ImageRenderLayer[] = [];
    for (const managedLayer of this.manager.rootLayers.managedLayers) {
      const userLayer = managedLayer.layer;
      if (userLayer === null || userLayer.type !== "image") continue;
      for (const renderLayer of userLayer.renderLayers) {
        if (renderLayer instanceof ImageRenderLayer) sources.push(renderLayer);
      }
    }
    const existing = this.imageSources.value;
    if (
      existing.length === sources.length &&
      sources.every((source, i) => source === existing[i])
    ) {
      return;
    }
    for (const source of sources) source.addRef();
    for (const source of existing) source.dispose();
    this.imageSources.value = sources;
  }

  restoreState(specification: any) {
    super.restoreState(specification);
    this.slice.restoreState(specification[SLICE_JSON_KEY]);
  }

  toJSON() {
    const x = super.toJSON();
    delete x[SOURCE_JSON_KEY];
    x[SLICE_JSON_KEY] = this.slice.toJSON();
    return x;
  }

  static type = "slice";
  static typeAbbreviation = "slc";
}

registerLayerType(SliceUserLayer);
