import { generatedLuts } from "#src/luts/generatedLuts.js";
export { generatedLuts };

type LutData = Record<string, number[]>;
type LutRgbStrings = Record<string, string[]>;

export const luts: LutData = Object.entries(generatedLuts).reduce(
  (accumulator, [key, value]) => ({
    ...accumulator,
    [key]: value.flat().map((item) => item / 255),
  }),
  {} as LutData
);

export const invertedLuts: LutData = Object.entries(generatedLuts).reduce(
  (accumulator, [key, value]) => ({
    ...accumulator,
    [key]: value
      .slice()
      .reverse()
      .flat()
      .map((item) => item / 255),
  }),
  {} as LutData
);

export const lutRgbStrings: LutRgbStrings = Object.entries(generatedLuts).reduce(
  (accumulator, [key, value]) => ({
    ...accumulator,
    [key]: value.map((c) => `rgb(${c.slice(0, 3).join()})`),
  }),
  {} as LutRgbStrings
);

export const lutRgbStringsInverted: LutRgbStrings = Object.entries(lutRgbStrings).reduce(
  (accumulator, [key, value]) => ({
    ...accumulator,
    [key]: value.slice().reverse(),
  }),
  {} as LutRgbStrings
);
