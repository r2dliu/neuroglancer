/**
 * @license
 * Copyright 2026 Google Inc.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import type { ChunkManager } from "#src/chunk_manager/frontend.js";
import {
  addUnconditionalSliceViews,
  CrossSectionSpecificationMap,
} from "#src/data_panel_layout.js";
import type { ManagedUserLayer, VisibleLayerInfo } from "#src/layer/index.js";
import { registerLayerType, UserLayer } from "#src/layer/index.js";
import type { PerspectivePanel } from "#src/perspective_view/panel.js";
import { PerspectiveViewRenderLayer } from "#src/perspective_view/render_layer.js";
import type { Borrowed, Owned } from "#src/util/disposable.js";
import { verifyObjectProperty } from "#src/util/json.js";

const CROSS_SECTIONS_JSON_KEY = "crossSections";

export class SliceRenderLayer extends PerspectiveViewRenderLayer {
  isTransparent = false;
  isAnnotation = false;
  isVolumeRendering = false;

  constructor(
    private chunkManager: Owned<ChunkManager>,
    public crossSections: Owned<CrossSectionSpecificationMap>,
  ) {
    super();
    this.registerDisposer(chunkManager);
    this.registerDisposer(crossSections);
    this.registerDisposer(
      crossSections.changed.add(this.redrawNeeded.dispatch),
    );
  }

  draw() {
    // The attached SliceView instances are rendered by PerspectivePanel.
  }

  attach(attachment: VisibleLayerInfo<PerspectivePanel>) {
    super.attach(attachment);
    const { view: panel } = attachment;
    addUnconditionalSliceViews(
      {
        chunkManager: this.chunkManager,
        layerManager: panel.viewer.layerManager,
        wireFrame: panel.viewer.wireFrame,
      },
      panel,
      this.crossSections,
      attachment,
    );
  }
}

export class SliceUserLayer extends UserLayer {
  crossSections: CrossSectionSpecificationMap;

  constructor(managedLayer: Borrowed<ManagedUserLayer>) {
    super(managedLayer);
    const { root } = managedLayer.manager;
    this.crossSections = this.registerDisposer(
      new CrossSectionSpecificationMap(
        root.navigationState.addRef(),
        root.crossSectionVoxelRange.addRef(),
      ),
    );
    this.registerDisposer(
      this.crossSections.changed.add(this.specificationChanged.dispatch),
    );
    this.addRenderLayer(
      new SliceRenderLayer(
        root.chunkManager.addRef(),
        this.crossSections.addRef(),
      ),
    );
  }

  canAddDataSource() {
    return false;
  }

  getDataSourceSpecifications(_layerSpec: any) {
    return [];
  }

  restoreState(specification: any) {
    super.restoreState(specification);
    this.crossSections.clear();
    verifyObjectProperty(specification, CROSS_SECTIONS_JSON_KEY, (value) => {
      if (value !== undefined) {
        this.crossSections.restoreState(value);
      }
    });
  }

  toJSON() {
    const specification = super.toJSON();
    specification[CROSS_SECTIONS_JSON_KEY] = this.crossSections.toJSON();
    return specification;
  }

  static type = "slice";
  static typeAbbreviation = "slc";
}

registerLayerType(SliceUserLayer);
