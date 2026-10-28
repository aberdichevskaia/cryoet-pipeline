# Instrument card: [microscope] + [detector]

One card per microscope and detector combination. Fill in what you know. Write
"unknown" rather than guess; if you do guess, say so. Any language is fine.

Filled in by: · Date: · Facility / site:

## Microscope and detector

- Microscope model and site:
- Detector model; counting or super-resolution mode:
- Energy filter and slit width (eV):
- Acceleration voltage (kV):
- Spherical aberration, Cs (mm):
- Phase plate (yes / no):

## Pixel size

For each magnification you use:

| Nominal magnification | Pixel size, physical (Å/px) | Super-resolution (Å/px) | How it was determined (calibration sample and date), or "nominal" |
|---|---|---|---|
| | | | |

- Do the file headers carry the right pixel size? If not, which value is right?

## Geometry

- Tilt-axis angle as the acquisition software reports it (degrees):
- Tilt-axis angle as found in practice, if different (for example, what alignment
  converges to):
- Which direction is positive tilt; any known handedness issue:
- Are images rotated or flipped relative to what the software reports?

## Gain reference

- Where it is stored, its file name and format:
- Are the frames already gain-corrected when they reach us?
- Rotation and flip of the gain relative to the frames, and in which program's
  convention (MotionCor, IMOD, RELION, Warp, …):
- How often a new gain reference is taken:

## Acquisition

- Software and version (SerialEM, Tomo, …):
- Tilt scheme (dose-symmetric, bidirectional, …), range and step:
- Dose per tilt and total dose (e⁻/Å²):
- Frames per tilt:
- Frame format (MRC, TIFF, EER) and compression:
- Target defocus range (µm):

## Files

- Folder layout, and one real file name per file type (frames, mdoc, gain,
  stacks):
- Are mdoc files written? Which of their fields can be trusted?
- What the file name encodes (series, tilt number, angle), and how: signs, leading
  zeros, how zero tilt is written:
- Anything the software writes wrongly into headers:

## Known quirks

- What newcomers to this instrument usually get wrong:
