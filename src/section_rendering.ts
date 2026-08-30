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

import { TrackableBoolean } from "#src/trackable_boolean.js";
import type { WatchableValueInterface } from "#src/trackable_value.js";
import { TrackableValue } from "#src/trackable_value.js";
import { RefCounted } from "#src/util/disposable.js";
import {
  verifyEnumString,
  verifyFiniteNonNegativeFloat,
  verifyObject,
} from "#src/util/json.js";
import { neverSignal, NullarySignal } from "#src/util/signal.js";
import type { Trackable } from "#src/util/trackable.js";
import { optionallyRestoreFromJsonMember } from "#src/util/trackable.js";
import { TrackableEnum } from "#src/util/trackable_enum.js";

export enum SectionRenderingMode {
  MIN = 0,
  MAX = 1,
}

export class TrackableSectionRenderingMode extends TrackableEnum<SectionRenderingMode> {
  constructor(value = SectionRenderingMode.MAX) {
    super(SectionRenderingMode, value);
  }
}

export interface SectionRenderingOptions {
  voxelRange: number;
  renderingMode: SectionRenderingMode;
  interactive: boolean;
}

export const defaultSectionRenderingOptions: SectionRenderingOptions = {
  voxelRange: 0,
  renderingMode: SectionRenderingMode.MAX,
  interactive: false,
};

export const defaultSectionRenderingOptionsWatchable: WatchableValueInterface<SectionRenderingOptions> =
  {
    changed: neverSignal,
    value: defaultSectionRenderingOptions,
  };

export function verifySectionRenderingMode(obj: any) {
  return verifyEnumString(obj, SectionRenderingMode);
}

export class SectionRenderingState
  extends RefCounted
  implements Trackable, WatchableValueInterface<SectionRenderingOptions>
{
  changed = new NullarySignal();
  voxelRange = new TrackableValue<number>(0, verifyFiniteNonNegativeFloat);
  renderingMode = new TrackableSectionRenderingMode();
  interactive = new TrackableBoolean(false);

  constructor() {
    super();
    for (const watchable of [
      this.voxelRange,
      this.renderingMode,
      this.interactive,
    ]) {
      watchable.changed.add(this.changed.dispatch);
    }
  }

  get value(): SectionRenderingOptions {
    return {
      voxelRange: this.voxelRange.value,
      renderingMode: this.renderingMode.value,
      interactive: this.interactive.value,
    };
  }

  restoreState(obj: any) {
    verifyObject(obj);
    optionallyRestoreFromJsonMember(obj, "voxelRange", this.voxelRange);
    optionallyRestoreFromJsonMember(obj, "renderingMode", this.renderingMode);
    optionallyRestoreFromJsonMember(obj, "interactive", this.interactive);
  }

  reset() {
    this.voxelRange.reset();
    this.renderingMode.reset();
    this.interactive.reset();
  }

  toJSON() {
    const voxelRange = this.voxelRange.toJSON();
    const renderingMode = this.renderingMode.toJSON();
    const interactive = this.interactive.toJSON();
    if (
      voxelRange === undefined &&
      renderingMode === undefined &&
      interactive === undefined
    ) {
      return undefined;
    }
    return {
      voxelRange,
      renderingMode,
      interactive,
    };
  }
}
