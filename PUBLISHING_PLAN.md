# Plan: GenerateDynamicalNetworks.jl as a publishable package

Status 2026-11-13: all decisions made. Steps 1 (scaffold) and 2 (core layer) are done and committed; step 3 is under way (CTRNN design ported). Step 1 was done in place: the prototype now lives in
`research/` (`research/benchmarks/`, `research/notebooks/`). Paths below that say `benchmarks/` or
`notebooks/` mean those.

## 0. Where things stand

- No `src/`, no git. About 4,600 lines of prototype code in `benchmarks/phase0–7/`, plus 8 Pluto
  notebooks built from `notebooks/tools/*_cells.jl` (1,300 lines). One environment, `notebooks/Project.toml`.
- The code is one include chain: phase2/horchler.jl → phase3/optimize.jl → phase4/backends.jl →
  phase5/backends.jl + classes.jl → phase6/synchrony.jl + seed1d.jl → phase7/hopfield.jl. Several later
  files redefine earlier names: `simulate_frames` per backend (fine as methods), `classify` in three files,
  `const OUT` in two.
- **The two unregistered path dependencies (checked 2026-11-13):**
  - **SHCTools:** the code that becomes the package calls exactly one function from it, `passagetime`
    (horchler.jl:122). That is the one-line closed-form Stone–Holmes leading-order mean:
    `(2(log δ − log ε) + log 4λu − log1p(λu/λs) + γ) / 2λu`. Everything else that uses SHCTools is
    research only: `phase0/` (`createcycle`, `lv_params`, `lv_integrate`) and `phase2/reference.jl` (the
    SHCTools notebook used as a reference).
  - **CTRNNs:** used only by `phase5/to_ctrnns.jl`, a one-off check that a fitted CTRNN behaves the same
    in CTRNNs.jl. No fit or backend needs it.
- **Hard-coded outside paths:** `phase2/reference.jl` and `phase5/worm_data.jl` (RobertsTracks). Both are
  research only.
- **Data:** 97 result CSVs (472 KB) and 15 MB of `.jls` checkpoints and run stores. The `.jls` files
  aren't kept: Serialization isn't stable across Julia versions.
- **How mature each backend is** (from the stored suites):

  | Backend | Result |
  |---|---|
  | CTRNN `:design` with relays | All 6 fixtures and the real worm (one branch on one check) |
  | Two-population | 10/12 |
  | Hopfield | 9/13 |
  | Synchrony oscillators | 8/13 old seed, 6/13 race seed |
  | Horchler LV (u form) | The only explicit dwell inversion; strong at low noise |
  | Simplex / cylinder | 13/24 together; cylinder worm ring fails outright |
  | CTRNN by CMA-ES | 0/6 (superseded by `:design`) |

## 1. Decisions

1. **Dependencies — decided:** neither is needed.
   - Inline `passagetime` into the Horchler backend, with a docstring citing Stone & Holmes (1990) and
     SHCTools.jl.
   - Drop the CTRNNs.jl conversion from v0.1. If interop is wanted later, add it as a package extension
     once CTRNNs.jl is registered.
   - Then GDN depends only on registered packages, and SHCTools.jl stays separate (as decided
     2026-11-06), just not a dependency.
   - The Phase 2 check that "the x form reproduces the SHCTools notebook" becomes a test against stored
     reference numbers.
2. **Backends — decided:** all ship in v0.1, each marked *stable* or *experimental* by its suite result.
3. **Lab notebooks — decided:** the prototype tree is frozen in `research/`, with the notebooks exported
   to static HTML. New docs and tutorials are written against the package.
4. **Data — decided: no real data.**
   - The package ships example networks only:
     - `ring(n)`
     - the six fixture graphs: foraging, figure-eight, Δ-clique, shared-solvable, shared-conflicting and
       the worm-shaped ring
     - the made-up two-way graphs: flip-flop, two-way-and-exit, worm-with-return, hub-with-two-returns
   - These go in `src/examples.jl` and are used by the tests and tutorials.
   - `research/` leaves out the worm-derived files (decided): `worm_data.csv`, `worm_data.jl` and the
     worm run stores (all `.jls` files are ignored anyway). Written: `research/DATA.md`. A `research/DATA.md` says how to get the data back:
     - **Source:** Roberts et al. (2016), "A stochastic neuronal model predicts random search behaviors
       at multiple spatial scales in C. elegans", eLife. Data on Dryad,
       [doi:10.5061/dryad.35qv6](https://doi.org/10.5061/dryad.35qv6) (listed as CC0).
     - **Files used:** `IBW files/C<n>_Viterbi.ibw` (the Viterbi-decoded HMM states) and
       `Wild type/C<n>/worm.ibw` (the frames per worm), for the wild-type cohorts C15, C21, C57, C162 and
       C186 (Table 1).
     - **How the graph was made:** states ordered Forward, Pause 1, Reverse, Pause 2 (Viterbi codes
       2, 0, 1, 3). A bout is an unbroken run of one state; each worm's first and last bouts are left out
       of the mean bout lengths; transitions are counted between consecutive bouts of the same worm.
       2 frames are dropped at each end of a worm, as in Roberts' `wormtracks.ipf`.
     - **Reader:** the prototype read the IBW files with a separate RobertsTracks.jl, which isn't part of
       this repo. The Table 1 means used for the Phase 3–4 worm ring are printed in the paper.
5. **Metadata — decided.**
   - Name `GenerateDynamicalNetworks` (repo `GenerateDynamicalNetworks.jl`), MIT licence.
   - `authors = ["denizhanpak <denpak@iu.edu>"]`, as in SHCTools.jl and CTRNNs.jl. The global git
     identity matches.
   - GitHub account `denizhanpak`; the repo and docs are public.
   - Not yet decided: CITATION.cff and a Zenodo DOI (default: CITATION.cff yes, Zenodo at the first
     release).

## 2. Target layout

```
GenerateDynamicalNetworks.jl/
  Project.toml                 # registered deps only, [compat] for each
  src/
    GenerateDynamicalNetworks.jl   # module, exports
    spec.jl         # TransitionSpec(P, τ; noise, sample), spec_problems, edge_list, successors (repair etc. go with Horchler)
    examples.jl     # ring(n), the fixture graphs, the two-way graphs
    simulate.jl     # one StochasticDiffEq EM + EnsembleThreads frame simulator; per-backend drift, noise, decoder
    measure.jl      # measure, verdict, score, validate (held-out sets, half step), acceptance report
    classify.jl     # one classify(model; ...) with backend methods; GHOST_RATIO etc. as keyword defaults
    fit/
      correct.jl    # per-state Broyden correction (one implementation for CTRNN, sync and Hopfield)
      seed.jl       # rough_rates, 1-D exit time, race seed (seed1d.jl), Stone–Holmes passage time
      search.jl     # CMA-ES search, polish (phase 3)
      batch.jl      # checkpoint / deadline / OutOfTime, chunked runs
    backends/
      horchler.jl  simplex.jl  cylinder.jl  twopop.jl  ctrnn.jl  sync.jl  hopfield.jl
  test/             # fast; see section 4
  benchmarks/       # the acceptance suites on the example networks as scripts on the package; results CSVs
  docs/             # Documenter + Literate: theory per backend, acceptance rule, tutorials, API, results
  research/         # frozen prototype tree (own environment, path deps allowed there), notebook HTML
  .github/workflows # CI (tests on 1.10 LTS + current), Documenter, CompatHelper, TagBot
```

The user-facing API, roughly:

```julia
spec  = TransitionSpec(P, τ; noise=0.1)   # or example(:foraging; noise=0.1)
model = fit(spec, CTRNNDesign())          # or TwoPopulation(), Synchrony(), Hopfield(), Horchler(:u), Simplex(), Cylinder()
report = validate(model, spec)            # dwell errors, branch CIs, off-graph, pass/fail on the agreed rule
classify(model, spec)                     # heteroclinic / excitable / ghost / limit cycle / unknown
simulate(model; T, seed)                  # decoded state sequence
```

Each backend is a struct holding its options. `fit` dispatches on it, and the model types are the existing
`CTRNNModel`, `SyncModel`, `HopModel`, …

## 3. Execution order

Each step ends with a checkpoint for you. Runs over 10 minutes are asked about first, as usual.

1. **Scaffold — done 2026-11-13.**
   - Generated with PkgTemplates in place: MIT; CI on 1.10, 1 and pre; Documenter; CompatHelper; TagBot;
     Dependabot; Aqua and JET tests.
   - The prototype moved into `research/`, with relative paths kept, and its suites still load.
   - `git init` and one local commit; nothing pushed.
2. **Port the core layer — done 2026-11-13 (not committed yet).**
   - `src/`: `spec.jl` (`TransitionSpec`, `spec_problems`, `edge_list`, `successors`), `examples.jl`
     (`ring`, `example`, `example_names`, `random_network`), `model.jl` (the backend interface,
     `AbstractNetworkModel`), `simulate.jl`, `measure.jl` (`measure`, `verdict`, `score`, `validate`,
     `accept`), `classify.jl`.
   - Depends on Graphs, SciMLBase and StochasticDiffEqLowOrder (just the EM solver), plus stdlibs.
   - Tests: 152 pass, including Aqua and JET, on Julia 1.13 (1.10 not available locally).
   - **Differential tests** (`test/differential/run.jl`, local only): 67 comparisons pass. They cover the
     examples, the random networks, and one model per backend (Horchler u and x, simplex, cylinder,
     two-population, CTRNN design, synchrony, Hopfield). Each backend's prototype model is wrapped in the
     new interface; runs, measurements, verdicts, scores, held-out checks and classes are identical to
     the prototype's. The wrappers are the starting point for step 3.
   - Changes from the prototype, on purpose:
     - The classifier counts a noise-free run that starts on an equilibrium as settled (`≤` instead of
       `<`); before, it came out "irregular".
     - `validate` defaults to T = 250Στ, as the suites used.
     - The graph repair and branch-noise rules (`repair`, `subdivide`, `branch_noise`) belong to the
       Horchler backend, so they move with it in step 3.
   - Found in the prototype: `classify(::NamedTuple)` always used the u-form drift, even for x-form
     models. It never met an x-form model, so no result is affected.
3. **Port the backends one at a time, in order of maturity:** CTRNN design with relays → two-population →
   synchrony with race seed → Hopfield → Horchler (with the inlined passage time) → simplex/cylinder.
   - Hopfield: the prototype's OSQP design is not always bit-reproducible across processes (seen 2026-11-13: the same
     fit gave a slightly different J and b in 5 of 16 builds, two variants in all, with visit counts 1–2% apart). The likely cause is OSQP's
     default adaptive step-size interval, which is set from measured setup time. Fix a fixed `adaptive_rho_interval` in the port.
   - **CTRNN design — ported 2026-11-13 (not committed yet).** `src/backends/ctrnn.jl`: `CTRNNModel`,
     `CTRNNDesign`, `fit`, checkpoint and deadline. `test/differential/ctrnn.jl` (176 checks) matches the prototype
     exactly for every example's layout, seed and design, for correction passes in both modes, for the fit pipeline
     and for the stored fits. A full refit of the stored two-way flip-flop fit reproduces its weights bit for bit
     (96 s). The CMA-ES CTRNN regimes (0/6) were not ported.
   - **Two-population — ported and committed 2026-11-13.** `src/backends/twopop.jl` plus the shared fitting code
     it needs: `src/fit/seed.jl` (Horchler dwell formulas, with the Stone–Holmes passage time inlined from SHCTools),
     `src/fit/edges.jl` (edge seed and correction, shared with simplex and cylinder) and `src/fit/search.jl` (CMA-ES
     on fresh noise). New deps: CMAEvolutionStrategy, Distributions, QuadGK. `test/differential/twopop.jl`: 97 checks
     identical, including the passage time against SHCTools itself and all 12 stored fits. The fit's search stops
     after 300 s by default (the prototype used 120 s; the user chose 300 s on 2026-11-13); `seconds=Inf` for
     machine-independent fits.
   - **Synchrony — ported and committed 2026-11-13.** `src/backends/synchrony.jl` (`SyncModel`, `Synchrony`,
     corner layout), `src/fit/correct.jl` (`rough_rates`, `initial_response`, `correct_rates`: the per-state Broyden
     correction, written so Hopfield can reuse it through `build`/`cap`) and `src/fit/race.jl` (the full-distribution
     race seed and response). `Synchrony(seed=:rough)` is the default, as in the prototype suite (8/13); `seed=:race`
     is the race seed (6/13). Dropped as research only: the pilot fit and 3-ring layout, the mean-only 1-D seed
     (`log_exit_time`, `seed_out_1d`, `race_response`), the unused `arrive`/`floor` options (every stored fit has 0) and
     `ω` (always 0). `test/differential/synchrony.jl`: 385 checks identical (layouts, rates, drift, race, race seeds for
     every example, correction passes and resume for both seeds, all 24 stored example fits rebuilt and simulated,
     and full refits of the stored Δ-clique 0.1 fits for both seeds); 10.6 min.
   - **Hopfield — ported 2026-11-13 (not committed yet).** `src/backends/hopfield.jl` (`HopModel`, `Hopfield`, layout
     search with one-hot fallback, OSQP design, race seed with per-edge contractions, the no-design fallback as
     `keep_last_design`). New deps JuMP 1.32.1 and OSQP 0.8.1, regular deps (user's choice). OSQP's
     `adaptive_rho_interval` is fixed at 25 (user's choice): repeated designs are identical and within 1e-9 of the old
     ones. Consequence: the stored Hopfield fits can't be reproduced bit for bit by anything, the prototype included
     (an unpatched refit of Δ-clique 0.1 gave rates up to 0.2% from the stored ones; the package's refit 0.18%).
     `test/differential/hopfield.jl` checks unpatched designs within 1e-8, then applies the fix to an in-memory copy of
     the prototype's `hop_design` and requires exact equality: layouts, designs, seeds and responses for every example,
     all 12 stored example fits rebuilt and simulated, a 4-pass fit with resume, and a full refit of Δ-clique 0.1
     (249 checks; ~16 min in total under a loaded machine). Without a checkpoint the package fit keeps a temporary one,
     so the fallback works either way (the prototype only fell back with a checkpoint). The stored shared-solvable 0.1
     fit (fell back after pass 5) predates the rate cap; refitted now it runs all 12 passes. Committed 2026-11-13.
   - **Horchler — ported 2026-11-13 (not committed yet).** `src/backends/horchler.jl`: graph repair (`repair`,
     `subdivide`, `branch_noise`), the closed-form design (`horchler_design`), `LVModel` (u and x forms; the x form's
     clamp is `project`; the type is shared with simplex next, user's choice) and `Horchler(form=:u; ...)`: dwell-request
     correction passes, held-out gate, CMA-ES on [log α; log ν] and two ν polish passes. The search stops after 300 s
     by default, as two-population (user's choice; the prototype had no time limit); `cma_search` gained `maxtime`.
     Renamed from the prototype: `design` → `horchler_design`, `correct` → `correct_requests`, `correct_rates` (ν
     polish) → `polish_saddles`, `raw_model` → `raw_design`. CI: the x-form design equals the SHCTools notebook's
     stored numbers to rounding; designed dwells, saddle eigenvalues, repair rules, a short fit. `test/differential/horchler.jl`:
     437 checks identical (repair and design for every example in both forms at three noise levels, drift, runs,
     classes, correction and polish passes, the search, a short fit, the stored worm-ring θs rebuilt, and full refits
     of all four stored phase 3 branching-suite cases reproducing their held-out numbers exactly); 9 min.
   Each one gets:
   - differential tests against the prototype
   - deterministic checks (design invariants, below)
   - regression tests that rebuild every stored fixture fit from the result CSVs and compare its design
     numbers
4. **Benchmarks.** Port the suite drivers, run on the example networks only, with results written as CSV.
   Rerun a reproduction subset (the pilot and one branching graph per backend) and compare against the
   stored tables. Exact equality is expected only where the RNG path is unchanged; elsewhere, agreement
   within the CIs. The run time gets measured and you're asked before the long part.
5. **Docs.**
   - Theory page per backend (equations, design, what fails and why), the acceptance rule and the
     held-out protocol.
   - Tutorials: 3-ring, a branching graph, a two-way graph with relays.
   - Results tables from the CSVs; API reference built from the docstrings, which mostly exist already.
   - The research notebooks as static HTML.
6. **Release.**
   - Aqua and JET clean; `[compat]` bounds taken from the working Manifest.
   - README with a minimal example and a backend-status table; CITATION.cff if wanted.
   - Push to GitHub, CI green, docs deployed, then register (`@JuliaRegistrator register`); Zenodo if
     wanted.
   - Every push and registration happens only when you ask.

## 4. Tests (fast enough for CI)

- **Spec checks:** the known-bad graphs from phase 1/2 are rejected or repaired as before.
- **`rough_rates` and the seeds:** values on the fixture graphs equal the stored contractions. All 13
  match today; checked 2026-11-13.
- **Design invariants, deterministic:**
  - Hopfield: every vertex's Jacobian eigenvalues equal the designed rates (the pilot's vertex check).
  - Synchrony: corner rates are as set.
  - CTRNN: the relay layout is right.
  - Horchler: the x/u design matches stored reference numbers from the SHCTools notebook.
- **1-D exit time and Stone–Holmes passage time:** against a small Monte Carlo, within its CI.
- **Decoders:** synthetic trajectories decode to the expected state sequences.
- **Fallbacks:** `NoDesign` handling (pass-1 error, keep the last pass) and resuming from a checkpoint,
  from `benchmarks/phase7/test_fallback.jl`.
- **One tiny end-to-end fit per stable backend** (3-ring, shortened T), checked loosely. The full
  acceptance stays in `benchmarks/`, not CI.

## 5. Known open research items (documented, not blockers)

- The acceptance rule (separate 95% CIs per branch, three checks) fails exact models by chance at the
  1-point level. That explains much of the 6–9/13 for sync and Hopfield, and is left as agreed.
- Hopfield shared-solvable at 0.1: state 6 is 22% long. Cylinder on the worm-shaped ring: undiagnosed.
- CTRNN by CMA-ES: 0/6, superseded by `:design`. Ship it as experimental, or drop it.
