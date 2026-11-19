import type { ManagedUserLayer, VisibleLayerInfo } from "#src/layer/index.js";
import { registerLayerType, UserLayer } from "#src/layer/index.js";
import { ObliqueSliceProjection } from "#src/layer/oblique_slice/base.js";
import {
  DisplayPose,
  NavigationState,
  OrientationState,
  Position,
  TrackableCrossSectionZoom,
} from "#src/navigation_state.js";
import type { PerspectivePanel } from "#src/perspective_view/panel.js";
import { PerspectiveViewRenderLayer } from "#src/perspective_view/render_layer.js";
import { SliceView } from "#src/sliceview/frontend.js";
import { WatchableValue } from "#src/trackable_value.js";
import type { Borrowed } from "#src/util/disposable.js";
import { quat, vec3 } from "#src/util/geom.js";

const MAX_SLICE_VIEW_PIXELS = 2048;
const SPATIAL_NAMES = ["x", "y", "z"];
const UNIT_X = vec3.fromValues(1, 0, 0);
const UNIT_Y = vec3.fromValues(0, 1, 0);
const UNIT_Z = vec3.fromValues(0, 0, 1);

class ObliqueSliceRenderLayer extends PerspectiveViewRenderLayer {
  constructor(public slicer: ObliqueSliceUserLayer) {
    super();
  }

  attach(attachment: VisibleLayerInfo<PerspectivePanel>) {
    const panel = attachment.view;
    const parent = panel.navigationState;
    const { slicer } = this;
    const displayInfo = parent.pose.displayDimensionRenderInfo;
    const position = new Position(parent.coordinateSpace);
    const orientation = new OrientationState();
    const zoom = new TrackableCrossSectionZoom(displayInfo.addRef());
    const navigationState = new NavigationState(
      new DisplayPose(position, displayInfo.addRef(), orientation),
      zoom,
      parent.depthRange.addRef(),
    );
    const sliceView = new SliceView(
      slicer.manager.chunkManager,
      panel.viewer.layerManager,
      navigationState,
      panel.viewer.wireFrame,
    );
    const clipBounds = { lower: vec3.create(), upper: vec3.create() };
    sliceView.clipBounds = clipBounds;

    const update = () => {
      const space = parent.coordinateSpace.value;
      if (!space.valid || space.bounds === undefined) return;
      const { displayDimensionIndices } = displayInfo.value;
      const global = Float32Array.from(parent.position.value);
      SPATIAL_NAMES.forEach((name, i) => {
        const dim = space.names.indexOf(name);
        if (dim !== -1) global[dim] = slicer.position.value[i];
      });
      const { lower, upper } = clipBounds;
      const p = vec3.create();
      for (let i = 0; i < 3; ++i) {
        const dim = displayDimensionIndices[i];
        if (dim === -1) return;
        p[i] = global[dim];
        lower[i] = space.bounds.lowerBounds[dim];
        upper[i] = space.bounds.upperBounds[dim];
      }

      const q = quat.normalize(
        quat.create(),
        slicer.orientation.value as unknown as quat,
      );
      const u = vec3.transformQuat(vec3.create(), UNIT_X, q);
      const v = vec3.transformQuat(vec3.create(), UNIT_Y, q);
      const n = vec3.transformQuat(vec3.create(), UNIT_Z, q);

      let uMin = Infinity;
      let uMax = -Infinity;
      let vMin = Infinity;
      let vMax = -Infinity;
      const a = vec3.create();
      const b = vec3.create();
      const hit = vec3.create();
      const offset = vec3.create();
      for (let corner = 0; corner < 8; ++corner) {
        for (let axis = 0; axis < 3; ++axis) {
          if (corner & (1 << axis)) continue;
          for (let i = 0; i < 3; ++i) {
            const high = (corner >> i) & 1;
            a[i] = high ? upper[i] : lower[i];
            b[i] = i === axis ? upper[i] : a[i];
          }
          const da = vec3.dot(n, vec3.subtract(offset, a, p));
          const db = vec3.dot(n, vec3.subtract(offset, b, p));
          if (da * db > 0 || da === db) continue;
          vec3.lerp(hit, a, b, da / (da - db));
          vec3.subtract(offset, hit, p);
          const su = vec3.dot(u, offset);
          const sv = vec3.dot(v, offset);
          uMin = Math.min(uMin, su);
          uMax = Math.max(uMax, su);
          vMin = Math.min(vMin, sv);
          vMax = Math.max(vMax, sv);
        }
      }
      if (uMin > uMax) {
        uMin = uMax = vMin = vMax = 0;
      }

      const center = vec3.scaleAndAdd(vec3.create(), p, u, (uMin + uMax) / 2);
      vec3.scaleAndAdd(center, center, v, (vMin + vMax) / 2);
      for (let i = 0; i < 3; ++i) {
        global[displayDimensionIndices[i]] = center[i];
      }
      position.value = global;
      quat.copy(orientation.orientation, q);
      orientation.changed.dispatch();

      const width = uMax - uMin;
      const height = vMax - vMin;
      const pixelSize = Math.max(
        1,
        Math.max(width, height) / MAX_SLICE_VIEW_PIXELS,
      );
      zoom.value = pixelSize;
      const pixelWidth = Math.max(1, Math.ceil(width / pixelSize));
      const pixelHeight = Math.max(1, Math.ceil(height / pixelSize));
      sliceView.projectionParameters.setViewport({
        width: pixelWidth,
        height: pixelHeight,
        logicalWidth: pixelWidth,
        logicalHeight: pixelHeight,
        visibleLeftFraction: 0,
        visibleTopFraction: 0,
        visibleWidthFraction: 1,
        visibleHeightFraction: 1,
      });
    };
    update();
    for (const signal of [
      slicer.position.changed,
      slicer.orientation.changed,
      parent.position.changed,
      parent.coordinateSpace.changed,
      displayInfo.changed,
    ]) {
      attachment.registerDisposer(signal.add(update));
    }

    panel.sliceViews.set(sliceView, true);
    attachment.registerDisposer(() => panel.sliceViews.delete(sliceView));
  }
}

export class ObliqueSliceUserLayer extends UserLayer {
  position = new WatchableValue<number[]>([0, 0, 0]);
  orientation = new WatchableValue<number[]>([0, 0, 0, 1]);
  voxelRange = new WatchableValue<number>(0);
  projection = new WatchableValue<ObliqueSliceProjection>(
    ObliqueSliceProjection.MAX,
  );

  constructor(managedLayer: Borrowed<ManagedUserLayer>) {
    super(managedLayer);
    this.addRenderLayer(new ObliqueSliceRenderLayer(this));
  }

  static type = "oblique_slice";
  static typeAbbreviation = "obl";
}

registerLayerType(ObliqueSliceUserLayer);
