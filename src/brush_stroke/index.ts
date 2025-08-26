/**
 * @license
 * Copyright 2024 Google Inc.
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

import { HashMapUint64 } from "#src/gpu_hash/hash_table.js";

/**
 * Simple hash table for storing brush stroke mask values at specific 3D coordinates.
 * This acts as a sparse 3D texture overlay.
 */
export class BrushHashTable extends HashMapUint64 {
    // Store coordinates for save functionality - maps hash key to [z, y, x]
    public coordinates = new Map<bigint, [number, number, number]>();
    private getBrushKey(z: number, y: number, x: number): bigint {
        const x1 = x >>> 0;
        const y1 = y >>> 0;
        const z1 = z >>> 0;

        const h1 = (((x1 * 73) * 1271) ^ ((y1 * 513) * 1345) ^ ((z1 * 421) * 675)) >>> 0;
        const h2 = (((x1 * 127) * 337) ^ ((y1 * 111) * 887) ^ ((z1 * 269) * 325)) >>> 0;

        return BigInt(h1) + (BigInt(h2) << 32n);
    }

    addBrushPoint(z: number, y: number, x: number, value: number) {
        const key = this.getBrushKey(z, y, x);
        this.delete(key);
        const brushValue = BigInt(value);
        this.set(key, brushValue);

        // Store coordinates for save functionality
        this.coordinates.set(key, [z, y, x]);
    }

    deleteBrushPoint(z: number, y: number, x: number) {
        const key = this.getBrushKey(z, y, x);
        this.delete(key);

        // Remove coordinates as well
        this.coordinates.delete(key);
    }

    getBrushValue(z: number, y: number, x: number): number | undefined {
        const key = this.getBrushKey(z, y, x);
        const value = this.get(key);
        if (value !== undefined) {
            return Number(value);
        }
        return undefined;
    }

    hasBrushPoint(z: number, y: number, x: number): boolean {
        const key = this.getBrushKey(z, y, x);
        return this.has(key);
    }

    /**
     * Override clear to also clear coordinates
     */
    clear() {
        this.coordinates.clear();
        return super.clear();
    }
}
