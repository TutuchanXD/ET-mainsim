# main_rd 1000x1000 Evaluation

This directory contains lightweight evaluation scripts for the ET main detector
simulation run. They do not replace the existing ET-mainsim production scripts.

The current experiment specification is:

- Detector: `main_rd`, 1000 x 1000 crop centered on the `main_rd` focal-plane center.
- Center: RA 304.41406499712303 deg, Dec 51.81987707392268 deg.
- Pixel: 10 um, 4.83 arcsec/pix.
- Cadence: 180 frames, 10 s per frame.
- Sky: 21 ET mag/arcsec^2 converted to electron/s/pix by Photsim7.
- Dark current: 1 e-/s/pix.
- Scattered light: disabled for this run.
- Readout noise: 6 e-/pix.
- Detector response: 1% inter-pixel and 1% intra-pixel response variation.
- Flat field and flat-field correction: disabled.
- ADC chain: full well 90680 e-, gain 1.4 e-/ADU, bias 3500 ADU,
  column noise sigma 5 ADU, 16-bit final ADC.
- Cosmic rays: enabled with `cosmic_ray/dark_test_10um/event_library_10um.npz`
  and rate 5 events cm^-2 s^-1.
- PSF: `241006/D280mm-focus`; smoke rendering chooses the nearest PSF field ID
  to the `main_rd` field angle.

Run star-count evaluation:

```bash
conda run --no-capture-output -n etbase \
  python /home/cxgao/ET/ET-mainsim/main_rd_1000_eval/count_main_rd_stars.py --save-csv
```

Run a dual-GPU one-frame smoke benchmark:

```bash
bash /home/cxgao/ET/ET-mainsim/main_rd_1000_eval/run_dual_gpu_smoke.sh
```

Outputs are written to:

```text
/home/cxgao/Results/ET-mainsim/main_rd_1000_eval
```

