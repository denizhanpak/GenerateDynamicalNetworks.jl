# Real data: not in this repository

This repository has no real data. The package, its tests and its benchmarks use only made-up example
networks. Phases 5–7 of the research prototype also fitted one graph measured from real worm tracks.
That graph (`benchmarks/phase5/results/worm_data.csv`) and the script that built it
(`benchmarks/phase5/worm_data.jl`) are left out. The result tables and logs from those fits are kept as
history, under the case name "worm data". Without the file, `worm_data_case()` returns `nothing`, and
the Phase 5–7 scripts that use the worm case can't run.

## Getting the data

Roberts, W. M. et al. (2016), "A stochastic neuronal model predicts random search behaviors at multiple
spatial scales in *C. elegans*", *eLife*. The tracking data are on Dryad:
[doi:10.5061/dryad.35qv6](https://doi.org/10.5061/dryad.35qv6), listed as CC0.

Files used, for each wild-type cohort in the paper's Table 1 (C15, C21, C57, C162 and C186):

- `IBW files/C<n>_Viterbi.ibw`: the most likely sequence of the four HMM states at each frame.
- `Wild type/C<n>/worm.ibw`: which worm each frame belongs to, used to split the Viterbi sequence by
  worm. Roberts' `wormtracks.ipf` drops 2 frames at each end of each worm, so each worm's frame count is
  its run length in `worm.ibw` minus 4. The prototype checks that these counts add up to the length of
  the Viterbi sequence.

The prototype read the Igor Binary Wave files with a separate reader, RobertsTracks.jl, which isn't part
of this repository. Any IBW reader will do.

## Rebuilding `worm_data.csv`

- **States:** Forward, Pause 1, Reverse, Pause 2, which are Viterbi codes 2, 0, 1 and 3.
- **Bouts:** a bout is an unbroken run of one state within one worm. Its length is the number of frames
  × the Viterbi wave's frame interval.
- **Transitions:** counted between consecutive bouts of the same worm. P is each row's counts divided by
  the row total.
- **Mean dwells and CVs:** use every bout except each worm's first and last, which the recording cuts off.

The file is CSV with the header `kind,from,to,count,value`. Its rows are:

- `transition,i,j,count,P_ij` for every transition seen
- `dwell,i,,n,mean bout length in s`
- `cv,i,,n,bout length CV`

`worm_data_case()` in `benchmarks/phase5/twoway_cases.jl` reads the `transition` and `dwell` rows.

The Phase 3–4 worm *ring* doesn't need the data: it uses the mean bout lengths printed in Roberts et al.
(2016), Table 1 (`benchmarks/phase3/worm.jl`).

## Other files not included

`benchmarks/phase5/classes.jl` and `benchmarks/phase5/precheck.jl` use one of the author's own 2-node
CTRNN fits from another project as a classifier test case. They read it from
`research/data/CTRNN2NeuronOpt.txt` (row 9: W column by column, then b, then log τ), which isn't
included. Without it, `fixtures()` and that precheck section can't run; everything else can.
