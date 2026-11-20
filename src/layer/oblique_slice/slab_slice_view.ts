import {
  type SlabLevelRule,
  type SlabMode,
  SlabRenderLayer,
} from "#src/layer/oblique_slice/slab_render_layer.js";
import type { FrontendTransformedSource } from "#src/sliceview/frontend.js";
import { SliceView } from "#src/sliceview/frontend.js";
import type {
  SliceViewRenderContext,
  SliceViewRenderLayer,
} from "#src/sliceview/renderlayer.js";
import { ImageRenderLayer } from "#src/sliceview/volume/image_renderlayer.js";
import type { RefCounted } from "#src/util/disposable.js";

interface SlabEntry {
  layer: SlabRenderLayer;
  owner: RefCounted;
  sources: { readonly value: FrontendTransformedSource[][] };
}

// The slicer's SliceView. With a slab set, image layers are drawn as the slab
// projection (beneath), and the remaining layers (segmentation) at the center
// plane on top. Without one it is a plain SliceView.
export class SlabSliceView extends SliceView {
  slab: { mode: SlabMode; halfThickness: number } | undefined = undefined;
  levelRule: SlabLevelRule = "volume";
  private slabLayers = new Map<ImageRenderLayer, SlabEntry>();

  setSlab(slab: SlabSliceView["slab"]) {
    const current = this.slab;
    if (
      current?.mode === slab?.mode &&
      current?.halfThickness === slab?.halfThickness
    ) {
      return;
    }
    this.slab = slab;
    if (slab === undefined) this.syncSlabLayers([]);
    this.viewChanged.dispatch();
  }

  protected shouldDrawLayer(renderLayer: SliceViewRenderLayer) {
    return (
      this.slab === undefined || !(renderLayer instanceof ImageRenderLayer)
    );
  }

  protected drawUnderlay(renderContext: SliceViewRenderContext) {
    const { slab } = this;
    if (slab === undefined) return 0;
    const images = this.visibleLayerList.filter(
      (layer) => layer instanceof ImageRenderLayer,
    ) as ImageRenderLayer[];
    this.syncSlabLayers(images);
    const { gl } = this;
    let count = 0;
    // Per channel: a depth-only pass leaves each pixel's best sample in the
    // depth buffer; a color pass then blends in only the fragment matching it,
    // with the channel's own blend mode.
    for (const image of images) {
      const entry = this.slabLayers.get(image);
      if (entry === undefined) continue;
      const draw = () =>
        entry.layer.draw(renderContext, entry.sources.value, slab.mode);
      gl.enable(WebGL2RenderingContext.DEPTH_TEST);
      gl.clearDepth(1);
      gl.clear(WebGL2RenderingContext.DEPTH_BUFFER_BIT);
      gl.disable(WebGL2RenderingContext.BLEND);
      gl.colorMask(false, false, false, false);
      gl.depthMask(true);
      gl.depthFunc(WebGL2RenderingContext.LESS);
      draw();
      gl.colorMask(true, true, true, true);
      gl.depthMask(false);
      gl.depthFunc(WebGL2RenderingContext.EQUAL);
      image.setGLBlendMode(gl, count);
      draw();
      gl.depthMask(true);
      gl.depthFunc(WebGL2RenderingContext.LESS);
      ++count;
    }
    return count;
  }

  private syncSlabLayers(images: ImageRenderLayer[]) {
    for (const [image, entry] of this.slabLayers) {
      if (images.includes(image)) continue;
      entry.owner.dispose();
      entry.layer.dispose();
      this.slabLayers.delete(image);
    }
    for (const image of images) {
      if (this.slabLayers.has(image)) continue;
      const layer = new SlabRenderLayer(image, this.levelRule);
      const { owner, sources } = layer.attachTo(this);
      owner.registerDisposer(
        layer.redrawNeeded.add(() => this.viewChanged.dispatch()),
      );
      this.slabLayers.set(image, { layer, owner, sources });
    }
  }

  disposed() {
    this.syncSlabLayers([]);
    super.disposed();
  }
}
