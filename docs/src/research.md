# Research record

The package was built from a research prototype, developed in phases. Each phase has a notebook that records
what was tried and what the runs showed. They are kept as a record: the code and numbers in them are the
prototype's, not the package's, and they aren't updated.

| Phase | Topic | Notebook |
|:--|:--|:--|
| 0 | Simulator and optimiser library benchmarks | [phase 0](notebooks/phase0_benchmarks.html) |
| 1 | Core layer: specification checks, graph repair, dwell measurement | [phase 1](notebooks/phase1_core.html) |
| 2 | Horchler Lotka–Volterra design | [phase 2](notebooks/phase2_horchler.html) |
| 3 | Fitting by search and correction | [phase 3](notebooks/phase3_optimize.html) |
| 4 | Simplex and cylinder constructions | [phase 4](notebooks/phase4_constructions.html) |

The pages are static: each shows the notebook's output as it was when exported, and its controls don't
respond. They load Pluto's scripts and styles from the web.

The notebooks themselves, including those of the later phases (classification and the two-population and CTRNN
backends, synchrony, Hopfield), are Pluto files in the repository's `research/notebooks/` folder, with the
prototype code and stored results beside them in `research/benchmarks/`.
