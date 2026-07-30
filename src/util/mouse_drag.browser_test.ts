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

import { describe, expect, it, vi } from "vitest";
import { startRelativeMouseDrag } from "#src/util/mouse_drag.js";

function makeInitialEvent(clientX: number, clientY: number) {
  return new MouseEvent("mousedown", {
    view: window,
    button: 0,
    clientX,
    clientY,
  });
}

function dispatchPointer(
  type: "pointermove" | "pointerup" | "pointercancel",
  clientX: number,
  clientY: number,
) {
  document.dispatchEvent(
    new PointerEvent(type, { button: 0, clientX, clientY }),
  );
}

describe("startRelativeMouseDrag", () => {
  it("reports the release-tail delta and removes every listener on pointerup", () => {
    const move = vi.fn();
    const finish = vi.fn();
    startRelativeMouseDrag(makeInitialEvent(10, 20), move, finish);

    dispatchPointer("pointermove", 14, 25);
    dispatchPointer("pointerup", 16, 28);

    expect(move).toHaveBeenCalledTimes(1);
    expect(move.mock.calls[0].slice(1)).toEqual([4, 5]);
    expect(finish).toHaveBeenCalledTimes(1);
    expect(finish.mock.calls[0].slice(1)).toEqual([2, 3]);

    dispatchPointer("pointermove", 30, 40);
    dispatchPointer("pointercancel", 30, 40);
    expect(move).toHaveBeenCalledTimes(1);
    expect(finish).toHaveBeenCalledTimes(1);
  });

  it("cleans up idempotently on pointercancel", () => {
    const move = vi.fn();
    const finish = vi.fn();
    startRelativeMouseDrag(makeInitialEvent(5, 7), move, finish);

    dispatchPointer("pointercancel", 8, 11);
    dispatchPointer("pointercancel", 9, 12);
    dispatchPointer("pointermove", 10, 13);
    dispatchPointer("pointerup", 10, 13);

    expect(move).not.toHaveBeenCalled();
    expect(finish).toHaveBeenCalledTimes(1);
    expect(finish.mock.calls[0].slice(1)).toEqual([3, 4]);
  });
});
