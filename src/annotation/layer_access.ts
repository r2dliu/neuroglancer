/**
 * @file Generic frontend accessors for an annotation layer's mutable source and
 * native display state.
 *
 * These reach into the annotation UserLayer's internal `annotationStates` /
 * `annotationDisplayState`, which application code should not poke directly.
 * They carry no app-specific semantics — any annotation layer can
 * use them.
 */

import type { Viewer } from "#src/viewer.js";

/** The layer's first writable annotation source, or undefined. */
export function getMutableAnnotationSource(userLayer: any): any | undefined {
  const states = userLayer?.annotationStates?.states;
  if (!states) return undefined;
  const state = states.find((s: any) => s.source && !s.source.readonly);
  return state?.source;
}

function writableAnnotationState(userLayer: any): any | undefined {
  const states = userLayer?.annotationStates?.states;
  if (!states) return undefined;
  return states.find((s: any) => s.source && !s.source.readonly) ?? states[0];
}

/**
 * Drive a layer's native annotation hover so an external highlight (e.g. a list
 * row) lights up the annotation exactly as if the cursor were over it.
 * `partIndex` 0 = the whole object (non-interactive). Pass null to clear.
 */
export function setAnnotationHover(
  viewer: Viewer,
  userLayer: any,
  id: string | null,
  partIndex = 0,
): void {
  const displayState = userLayer?.annotationDisplayState;
  if (!displayState?.hoverState) return;
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
  (viewer as any).display?.scheduleRedraw?.();
}

/**
 * Reflect an externally-owned selection into a layer's native
 * `selectedAnnotation`. `controlledSelection` gates neuroglancer's hover/pick so
 * it can't clobber the externally-owned value. Pass null to deselect.
 */
export function setAnnotationSelection(
  userLayer: any,
  id: string | null,
): void {
  const displayState = userLayer?.annotationDisplayState;
  if (!displayState?.selectedAnnotation) return;
  displayState.controlledSelection = true;
  displayState.selectedAnnotation.value = id ?? undefined;
}
