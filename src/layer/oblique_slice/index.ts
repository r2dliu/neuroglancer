import { registerLayerType, UserLayer } from "#src/layer/index.js";
import { WatchableValue } from "#src/trackable_value.js";

export enum ObliqueSliceProjection {
  NONE = 0,
  MIN = 1,
  MAX = 2,
}

export class ObliqueSliceUserLayer extends UserLayer {
  position = new WatchableValue<number[]>([0, 0, 0]);
  orientation = new WatchableValue<number[]>([0, 0, 0, 1]);
  voxelRange = new WatchableValue<number>(0);
  projection = new WatchableValue<ObliqueSliceProjection>(
    ObliqueSliceProjection.MAX,
  );

  static type = "oblique_slice";
  static typeAbbreviation = "obl";
}

registerLayerType(ObliqueSliceUserLayer);
