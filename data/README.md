# Dataset layout

Place unpaired RGB images in:

- `trainA/` — source domain A (for example, horses)
- `trainB/` — source domain B (for example, zebras)
- `testA/` — held-out source-domain images for A→B translation and cycle checks
- `testB/` — held-out target-domain images for B→A translation and cycle checks

Supported extensions are JPG, JPEG, PNG, BMP, and WebP. Images are not included because dataset licenses and sizes vary. The training loader randomly pairs only `trainA`/`trainB`, resizes, crops, flips, and normalizes each sample. Evaluation never mixes held-out images into training; it translates each test image in both the forward and cycle-reconstruction paths.
