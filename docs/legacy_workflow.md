# Legacy Full-Effect Workflow

`legacy-sim` preserves the active full profile of the removed
`et_sim_100_det.py` while delegating its implementation to Photsim7. Ray is
local only; no hostname or remote-address fallback exists.

The production preset uses 101 x 101 images, 360 ten-second cadences, one run
of 100 targets, seeded ET magnitudes from 7 to 17, G/ET background limit 17,
100 jitter-integrated PSFs with 300 samples each, one local CUDA Ray actor, and
`store_images=false`. The smoke preset uses 9 x 9, two cadences, one target,
and one CPU actor. Both intended-runnable defaults use the same 100-model,
three-axis, 300-sample native ET bank contract; smoke reduces the image and
cadence workload, not the scientific bank dimensions.

With the default 100 x 300 request, the compatibility contract binds
`jitter/et/native/legacy_science_v1_et_attitude_xyz_100x3x300_v1.npy` and its
strict manifest. Their accepted hashes are verified before NumPy load, and the
bank is fixed for the observation; per-cadence model choice comes from the
Photsim7 selection contract rather than `frame_index % n_models`.

The authoritative manifest explicitly records 23 enabled effects: target and
background scenes; stellar/background/scattered/dark/readout noise; scripted
and both whole-pixel gains; ET attitude motion, DVA, thermal drift and momentum
dump; jitter-integrated PSF and breathing; three pixel-response effects; coadd,
Kepler OA and OA helpers. Flat field, pixel-flux filtering, transit injection,
and cosmic rays remain disabled for historical parity.

The removed three-day `main_rd_reference` ramp-and-reset profile is not a
production science breathing model. ET full-frame/stamp production use the
shared TESS temperature timeline and the approved temperature-to-PSF-scale
relation. The named legacy compatibility profile retains its explicit legacy
breathing mode for reconstructability; that compatibility output must not be
presented as proof that the complete `legacy_science_v1` baseline has been met.

Every `legacy/run_N/` must contain all pickle/OA products plus
`legacy_effect_manifest.json`. ET-mainsim deserializes the products, validates
array ranks and target counts, and verifies the 23/4 effect inventory before a
run is complete. Resume skips only when every requested run passes this
readback. Partial output is never resumed because the underlying historical
Simulator partial-resume path is not reliable.

The verified Stage 2 bank and selection identities close only the discrete
PSF/JI selection sub-gate. Full image crop/normalization goldens, ensemble
tolerances, reduction products, and final RMS/CDPP comparison remain separate
acceptance work. See
[Stage 2 geometry, PSF, and jitter selection truth](stage2_selection_truth.md).
