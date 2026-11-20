import type { ManagedUserLayer, VisibleLayerInfo } from "#src/layer/index.js";
import { registerLayerType, UserLayer } from "#src/layer/index.js";
import { ObliqueSliceProjection } from "#src/layer/oblique_slice/base.js";
import { capPolygon, slicerFrame } from "#src/layer/oblique_slice/frame.js";
import {
  GizmoPart,
  ObliqueSliceGizmoRenderLayer,
} from "#src/layer/oblique_slice/gizmo.js";
import { SlabSliceView } from "#src/layer/oblique_slice/slab_slice_view.js";
import {
  DisplayPose,
  NavigationState,
  OrientationState,
  Position,
  TrackableCrossSectionZoom,
  TrackableDepthRange,
} from "#src/navigation_state.js";
import type { PerspectivePanel } from "#src/perspective_view/panel.js";
import { PerspectiveViewRenderLayer } from "#src/perspective_view/render_layer.js";
import type { WatchableValueInterface } from "#src/trackable_value.js";
import { WatchableValue } from "#src/trackable_value.js";
import type { Borrowed } from "#src/util/disposable.js";
import { quat, vec3 } from "#src/util/geom.js";
import { NullarySignal } from "#src/util/signal.js";

const MAX_SLICE_VIEW_PIXELS = 4096;

class ObliqueSliceRenderLayer extends PerspectiveViewRenderLayer {
  constructor(
    public slicer: ObliqueSliceUserLayer,
    private gizmo: ObliqueSliceGizmoRenderLayer,
  ) {
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
    // In slab mode the SliceView's orthographic camera spans the slab.
    const depthRange = new TrackableDepthRange(1, displayInfo);
    const navigationState = new NavigationState(
      new DisplayPose(position, displayInfo.addRef(), orientation),
      zoom,
      depthRange,
    );
    const sliceView = new SlabSliceView(
      slicer.manager.chunkManager,
      panel.viewer.layerManager,
      navigationState,
      panel.viewer.wireFrame,
    );
    const clipBounds = { lower: vec3.create(), upper: vec3.create() };
    sliceView.clipBounds = clipBounds;
    const updatePickTarget = () => {
      sliceView.pickTarget = slicer.selected.value
        ? { renderLayer: this.gizmo, part: GizmoPart.PLANE }
        : undefined;
    };
    updatePickTarget();
    attachment.registerDisposer(slicer.selected.changed.add(updatePickTarget));

    const updateSlab = () => {
      const halfThickness = slicer.voxelRange.value;
      const projection = slicer.projection.value;
      depthRange.value = Math.max(1, halfThickness);
      sliceView.setSlab(
        halfThickness > 0 && projection !== ObliqueSliceProjection.NONE
          ? {
              mode: projection === ObliqueSliceProjection.MIN ? "min" : "max",
              halfThickness,
            }
          : undefined,
      );
    };
    updateSlab();
    attachment.registerDisposer(slicer.voxelRange.changed.add(updateSlab));
    attachment.registerDisposer(slicer.projection.changed.add(updateSlab));

    const update = () => {
      const frame = slicerFrame(
        {
          position: slicer.position.value,
          orientation: slicer.orientation.value,
        },
        parent,
      );
      if (frame === undefined) return;
      vec3.copy(clipBounds.lower, frame.lower);
      vec3.copy(clipBounds.upper, frame.upper);

      let uMin = 0;
      let uMax = 0;
      let vMin = 0;
      let vMax = 0;
      const offset = vec3.create();
      capPolygon(frame, 0).forEach((p, i) => {
        vec3.subtract(offset, p, frame.center);
        const su = vec3.dot(frame.u, offset);
        const sv = vec3.dot(frame.v, offset);
        uMin = i === 0 ? su : Math.min(uMin, su);
        uMax = i === 0 ? su : Math.max(uMax, su);
        vMin = i === 0 ? sv : Math.min(vMin, sv);
        vMax = i === 0 ? sv : Math.max(vMax, sv);
      });

      const center = vec3.scaleAndAdd(
        vec3.create(),
        frame.center,
        frame.u,
        (uMin + uMax) / 2,
      );
      vec3.scaleAndAdd(center, center, frame.v, (vMin + vMax) / 2);
      const { global } = frame;
      for (let i = 0; i < 3; ++i) global[frame.displayDims[i]] = center[i];
      position.value = global;
      quat.copy(orientation.orientation, frame.rotation);
      orientation.changed.dispatch();

      const width = uMax - uMin;
      const height = vMax - vMin;
      // The nav panels' pixel size, so the slicer samples the same resolution
      // (and chunks) as the 2D views, capped to a GPU-safe texture.
      const pixelSize = Math.max(
        slicer.crossSectionZoom?.value ?? 1,
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
    if (slicer.crossSectionZoom !== undefined) {
      attachment.registerDisposer(slicer.crossSectionZoom.changed.add(update));
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
  selected = new WatchableValue<boolean>(false);
  // The nav panels' zoom (the viewer's cross-section scale), set by the app
  // before the layer is added; the slab samples at that resolution.
  crossSectionZoom: WatchableValueInterface<number> | undefined = undefined;
  // Fires when a gizmo drag ends, so the app can save the new pose.
  committed = new NullarySignal();

  constructor(managedLayer: Borrowed<ManagedUserLayer>) {
    super(managedLayer);
    const gizmo = new ObliqueSliceGizmoRenderLayer(this);
    this.addRenderLayer(new ObliqueSliceRenderLayer(this, gizmo));
    this.addRenderLayer(gizmo);
  }

  static type = "oblique_slice";
  static typeAbbreviation = "obl";
}

registerLayerType(ObliqueSliceUserLayer);
