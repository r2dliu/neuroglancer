/**
 * @file Generic frontend accessors for an annotation layer's mutable source and
 * native display state.
 *
 * These reach into the annotation UserLayer's internal `annotationStates` /
 * `annotationDisplayState`, which application code should not poke directly.
 * They carry no app-specific semantics — any annotation layer can
 * use them.
 */

import type { AnnotationLayerState } from "#src/annotation/annotation_layer_state.js";
import type { MultiscaleAnnotationSource } from "#src/annotation/frontend_source.js";
import type { AnnotationSource } from "#src/annotation/index.js";
import type { UserLayerWithAnnotations } from "#src/ui/annotations.js";
import type { Viewer } from "#src/viewer.js";

type AnnotationLayer = UserLayerWithAnnotations | null | undefined;

/** The layer's first writable annotation source, or undefined. */
export function getMutableAnnotationSource(
  userLayer: AnnotationLayer,
): AnnotationSource | MultiscaleAnnotationSource | undefined {
  return userLayer?.annotationStates.states.find((s) => !s.source.readonly)
    ?.source;
}

function writableAnnotationState(
  userLayer: AnnotationLayer,
): AnnotationLayerState | undefined {
  const states = userLayer?.annotationStates.states;
  if (states === undefined) return undefined;
  return states.find((s) => !s.source.readonly) ?? states[0];
}

/**
 * Drive a layer's native annotation hover so an external highlight (e.g. a list
 * row) lights up the annotation exactly as if the cursor were over it.
 * `partIndex` 0 = the whole object (non-interactive). Pass null to clear.
 */
export function setAnnotationHover(
  viewer: Viewer,
  userLayer: AnnotationLayer,
  id: string | null,
  partIndex = 0,
): void {
  const displayState = userLayer?.annotationDisplayState;
  if (displayState === undefined) return;
  if (id === null) {
    displayState.hoverState.value = undefined;
  } else {
    const state = writableAnnotationState(userLayer);
    if (!state) return;
    displayState.hoverState.value = {
      id,
      partIndex,
      annotationLayerState: state,
    };
  }
  viewer.display.scheduleRedraw();
}

/**
 * Reflect an externally-owned selection into a layer's native
 * `selectedAnnotation`. `controlledSelection` gates neuroglancer's hover/pick so
 * it can't clobber the externally-owned value. Pass null to deselect.
 */
export function setAnnotationSelection(
  userLayer: AnnotationLayer,
  id: string | null,
): void {
  const displayState = userLayer?.annotationDisplayState;
  if (displayState === undefined) return;
  displayState.controlledSelection = true;
  displayState.selectedAnnotation.value = id ?? undefined;
}
