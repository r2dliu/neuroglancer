import { debounce } from "lodash-es";
import type { PaneOverride, ViewerUIState } from "#src/data_panel_layout.js";
import { getCommonSliceViewerState } from "#src/data_panel_layout.js";
import {
  displayToSlicerPosition,
  slicerFrame,
} from "#src/layer/oblique_slice/frame.js";
import type { ObliqueSliceUserLayer } from "#src/layer/oblique_slice/index.js";
import { bindSlab } from "#src/layer/oblique_slice/index.js";
import { SlabSliceView } from "#src/layer/oblique_slice/slab_slice_view.js";
import {
  DisplayPose,
  NavigationState,
  OrientationState,
  Position,
  TrackableCrossSectionZoom,
  TrackableDepthRange,
} from "#src/navigation_state.js";
import { SliceViewPanel } from "#src/sliceview/panel.js";
import { RefCounted } from "#src/util/disposable.js";
import { quat, vec3 } from "#src/util/geom.js";

const COMMIT_DELAY_MS = 400;

// A 2D pane showing the slicer's plane (and slab). Its orientation follows the
// slicer and its position stays on the plane: moving along the normal (scroll,
// keys) moves the slicer, committed once the gesture settles; panning within
// the plane and zoom are the pane's own.
export function makeSlicerPane(slicer: ObliqueSliceUserLayer): PaneOverride {
  return (element, viewer) => new SlicerPane(slicer, element, viewer);
}

class SlicerPane extends RefCounted {
  constructor(
    slicer: ObliqueSliceUserLayer,
    element: HTMLElement,
    viewer: ViewerUIState,
  ) {
    super();
    const parent = viewer.navigationState;
    const displayInfo = parent.pose.displayDimensionRenderInfo;
    const position = new Position(parent.coordinateSpace);
    const orientation = new OrientationState();
    const zoom = new TrackableCrossSectionZoom(displayInfo.addRef());
    zoom.value = parent.zoomFactor.value;
    const depthRange = new TrackableDepthRange(1, displayInfo);
    const navigationState = new NavigationState(
      new DisplayPose(position, displayInfo.addRef(), orientation),
      zoom,
      depthRange,
    );
    const sliceView = new SlabSliceView(
      viewer.chunkManager,
      viewer.layerManager,
      navigationState,
      viewer.wireFrame,
    );
    sliceView.levelRule = "slice";
    this.registerDisposer(bindSlab(slicer, sliceView, depthRange));
    this.registerDisposer(
      new SliceViewPanel(viewer.display, element, sliceView, {
        ...getCommonSliceViewerState(viewer),
        navigationState,
        showScaleBar: viewer.showScaleBar,
      }),
    );

    const frameOf = () =>
      slicerFrame(
        {
          position: slicer.position.value,
          orientation: slicer.orientation.value,
        },
        parent,
      );
    const point = vec3.create();
    const offset = vec3.create();
    let placed = false;
    let syncing = false;

    const followSlicer = () => {
      const frame = frameOf();
      if (frame === undefined) return;
      syncing = true;
      try {
        if (!quat.equals(orientation.orientation, frame.rotation)) {
          quat.copy(orientation.orientation, frame.rotation);
          orientation.changed.dispatch();
        }
        if (placed) {
          const coordinates = position.value;
          for (let i = 0; i < 3; ++i) {
            point[i] = coordinates[frame.displayDims[i]];
          }
          vec3.subtract(offset, point, frame.center);
          vec3.scaleAndAdd(point, point, frame.n, -vec3.dot(frame.n, offset));
        } else {
          vec3.copy(point, frame.center);
          placed = true;
        }
        const { global } = frame;
        for (let i = 0; i < 3; ++i) global[frame.displayDims[i]] = point[i];
        position.value = global;
      } finally {
        syncing = false;
      }
    };

    const commit = this.registerCancellable(
      debounce(() => slicer.committed.dispatch(), COMMIT_DELAY_MS),
    );

    const moveSlicer = () => {
      if (syncing) return;
      const frame = frameOf();
      if (frame === undefined) return;
      const coordinates = position.value;
      for (let i = 0; i < 3; ++i) {
        point[i] = coordinates[frame.displayDims[i]];
      }
      vec3.subtract(offset, point, frame.center);
      const distance = vec3.dot(frame.n, offset);
      if (Math.abs(distance) < 1e-6) return;
      const center = vec3.scaleAndAdd(
        vec3.create(),
        frame.center,
        frame.n,
        distance,
      );
      slicer.position.value = displayToSlicerPosition(
        frame,
        parent,
        center,
        slicer.position.value,
      );
      commit();
    };

    const rotateSlicer = () => {
      if (syncing) return;
      slicer.orientation.value = Array.from(orientation.orientation);
      commit();
    };

    followSlicer();
    for (const signal of [
      slicer.position.changed,
      slicer.orientation.changed,
      parent.coordinateSpace.changed,
      displayInfo.changed,
    ]) {
      this.registerDisposer(signal.add(followSlicer));
    }
    this.registerDisposer(position.changed.add(moveSlicer));
    this.registerDisposer(orientation.changed.add(rotateSlicer));
  }
}
