# Cold-image implementation clarification

Recorded after the first DomainNet clipart/hash quality units completed, but
before any cold-image timing. The frozen target, methods, dimensions, seed,
worker count and repetition count remain as specified in ExecutionDetails.md.

Each selector must encode the images it actually uses. MMD50 uses 7680 source,
512 target-screen and 1024 target-validation images (9216 total). Sequential25,
batch25 and full455 do not read target-screen features; each encodes 8704 images.
This corrects an implementation that would otherwise charge all methods for an
unused target-screen stage. It is a measurement-scope correction, not a policy
or hyperparameter change. The code hash is saved with each cold result.

Model loading, image I/O/transforms, GPU extraction, projection and decision
execution are measured in the same process. Raw image downloads, checkpoint
downloads, Python imports and controlled OS cache eviction remain outside the
claim. The single repetition per method cannot establish a timing confidence
interval or represent all six target domains.
