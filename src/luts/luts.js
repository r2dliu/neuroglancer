// eslint-disable-next-line no-restricted-imports
import { generatedLuts } from "./generatedLuts.js";
export { generatedLuts };

export const luts = Object.entries(generatedLuts).reduce(
  (accumulator, [key, value]) => ({
    ...accumulator,
    [key]: value.flat().map((item) => item / 255),
  }),
  {}
);

export const invertedLuts = Object.entries(generatedLuts).reduce(
  (accumulator, [key, value]) => ({
    ...accumulator,
    [key]: value
      .slice()
      .reverse()
      .flat()
      .map((item) => item / 255),
  }),
  {}
);

export const lutRgbStrings = Object.entries(generatedLuts).reduce(
  (accumulator, [key, value]) => ({
    ...accumulator,
    [key]: value.map((c) => `rgb(${c.slice(0, 3).join()})`),
  }),
  {}
);

export const lutRgbStringsInverted = Object.entries(lutRgbStrings).reduce(
  (accumulator, [key, value]) => ({
    ...accumulator,
    [key]: value.slice().reverse(),
  }),
  {}
);
