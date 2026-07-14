# Historical artifact readback

`export_last90_truth_tables.py` is a read-only compatibility tool for finished
main-RD `*last90` runs. It does not define a maintained simulation workflow and
must not be used to resume or append to historical output directories.

For current runs, prefer the resolved `SimulationSpec`, catalog cache,
`effects_timeseries` metadata, and product schemas recorded by `et-mainsim`.
