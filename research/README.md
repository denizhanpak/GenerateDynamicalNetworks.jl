# Research prototype (frozen)

This folder is the phase-by-phase prototype that GenerateDynamicalNetworks.jl is being built from. It is
kept as a record of the experiments and results. It isn't part of the package and isn't maintained; the
package in `../src` supersedes it as backends are ported.

## Layout

- `notebooks/phase<k>_*.jl`: one Pluto notebook per phase. Each reads its results from
  `benchmarks/phase<k>/results/`. They are built from `notebooks/tools/phase<k>_cells.jl` by
  `tools/build_notebook.jl`.
- `benchmarks/phase<k>/`: the code and scripts for each phase. Later phases `include` earlier ones, in a
  chain from phase 2 to phase 7.
- `benchmarks/phase<k>/results/`: result CSVs and logs. Serialized checkpoints (`.jls`) aren't kept.

| Phase | Topic |
|---|---|
| 0 | Simulator and optimiser library benchmarks |
| 1 | Core layer: spec checks, graph repair, dwell measurement |
| 2 | Horchler Lotka–Volterra design (x and u forms) |
| 3 | Fitting by CMA-ES search and correction |
| 4 | Simplex and cylinder constructions |
| 5 | Dynamical classification; two-population and CTRNN backends (CTRNN design with relay cells) |
| 6 | Phase oscillators with the network in synchrony space |
| 7 | Hopfield (Chossat–Krupa) networks |

## Running

Run scripts from `notebooks/` with `julia --project=. -t <threads> ../benchmarks/phase<k>/<script>.jl`.
The environment refers to two unregistered packages by relative path, so to run it you need them checked
out beside this repository:

- SHCTools.jl at `../../../SHCTools.jl`
- CTRNNs.jl at `../../../CTRNNs.jl`

The package itself depends on neither.

Real data used in phases 5–7 isn't included; see [DATA.md](DATA.md) for how to download and rebuild it.
