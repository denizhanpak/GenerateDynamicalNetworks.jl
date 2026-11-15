```@meta
CurrentModule = GenerateDynamicalNetworks
```

# [CTRNN design](@id ctrnn-backend)

!!! note "Status: stable"
    Passes the acceptance protocol on most of the benchmark cases; see
    [Benchmark results](@ref ctrnn-results) below.

A continuous-time recurrent neural network with one cell per state. While one cell is on, noise fires one of
the cells that may follow it, and the cell that fires switches the rest off. Every edge has one weight, solved
from a one-dimensional first-passage time. There is no search: the design is followed by correction passes
only. Two-way edges are supported through hidden relay cells.

```julia
model = fit(spec, CTRNNDesign())
```

## [The system](@id ctrnn-system)

```math
dy_i = \frac{1}{\tau_i}\Big(-y_i + \sum_j W_{ij}\,\sigma\big(g\,(y_j + b_j)\big)\Big)dt + \varepsilon\,dW_i ,
\qquad \sigma(x) = \frac{1}{1 + e^{-x}} .
```

The first cells are the states, and any after them are relay cells. A frame is read as the state cell with the
largest ``g\,(y_i + b_i)``; relay cells are never reported.

## The design

Levels are set in units of a cell's own noise. A cell left alone fluctuates with standard deviation
``s_i = \varepsilon\sqrt{\tau_i / 2}``.

| Quantity | Value |
|:--|:--|
| time constant ``\tau_i`` | a quarter of the shortest dwell among state ``i`` and the states that lead to it |
| self-weight ``W_{ii} = A_i`` | ``12\,s_i``: the level a cell holds when it is on |
| inhibition | ``-1.5\,A_i`` from every other cell, unless an edge says otherwise |
| gain ``g`` | 2 divided by the smallest ``s_i`` |
| biases ``b_i`` | 0 |

### One weight per edge

While cell ``i`` is on, a cell ``j`` that can follow it is a one-dimensional noisy unit,

```math
\tau_j\,dy_j = \big(-y_j - d_{ij} + A_j\,\sigma(g\,y_j)\big)dt + \tau_j\,\varepsilon\,dW ,
```

held at an offset ``d_{ij}`` by the weight ``W_{ji} = -d_{ij}``. The mean time for noise to carry it up to
``A_j / 2``, where its self-weight takes over, is a one-dimensional first-passage integral
([`ctrnn_passage_time`](@ref)). The offset is solved so that this time is the edge's share of the dwell:
the edge ``i \to j`` should fire at the rate ``P_{ij} / \tau_i``.

The rates of a state's successors add up. Their sum sets the dwell and their ratios set the branch
probabilities, so one set of weights gives both.

### Relay cells

In a two-way pair ``i \leftrightarrow j`` one weight would have two jobs. ``W_{ij}`` would have to be strongly
negative, so that ``j`` switches ``i`` off when it fires, and close to zero, so that noise can fire ``i`` while
``j`` is on. So one direction of each pair goes through a hidden relay cell: while ``j`` is on, noise fires
the relay as it would any other successor; the relay drives ``i`` on over ``j``'s inhibition and holds the
other states off; and ``i`` then switches the relay off. A graph without two-way edges gets no relay.

```@example ctrnn
using GenerateDynamicalNetworks
const GDN = GenerateDynamicalNetworks

layout = GDN.ctrnn_layout(example("two-way and exit"; noise=0.1))
layout.n, layout.rel
```

Three state cells, and one relay for the edge 2 → 1. [A two-way graph](@ref) shows a relay at work.

## The fit

The designed offsets are corrected from simulation ([`ctrnn_correct_offsets`](@ref)).

- Each pass simulates the current design on new seeds: 32 runs of 250 times the sum of the dwells. The last
  three passes use `late_runs` runs at half the step.
- For each state, the pass compares the measured exit rate and branch ratios with their targets and takes one
  Broyden step on the state's total rate and rate ratios. How the measurements respond to the rates is learned
  from pass to pass, starting from the identity.

A relay's handoff takes a nearly fixed time, so part of a dwell doesn't respond to the weights; that is why
the response is learned and not assumed.

| Option | Default | Meaning |
|:--|:--|:--|
| `by_state` | `true` | correct state by state as above; `false` moves each edge by its own rate error |
| `passes` | 10 (8 with `by_state=false`) | correction passes |
| `late_runs` | 192 (96 with `by_state=false`) | runs in each of the last three passes |
| `base` | 100 000 | first seed; different values give independent fits |

A fit can be run in batches: `fit(spec, CTRNNDesign(); checkpoint=file, deadline=time() + 600)` saves its
progress to `file` and throws [`OutOfTime`](@ref) when it expects to pass the deadline. Calling it again with
the same file resumes.

## Classes

The class depends on the fit. Where every successor is held below its threshold, each state is a stable
equilibrium and the model is excitable. An offset can also come out on the other side, so that a successor
rises without any noise; a model with a loop of such transitions runs on its own. In the benchmark suite seven
of the ten fits are excitable, two are ghosts and one is unknown ([Results](@ref)).

## Limits

- **Only mean dwells are fitted.** A handoff through a relay is close to deterministic, so states that are
  left through a relay have more regular dwells than a noise-driven exit gives.
- **Branch checks.** The two failures in the benchmark suite are single branch checks, each on one set of runs.
- **High barriers.** The first-passage integral is evaluated in plain floating point. It is accurate at the
  design's levels but can overflow for cells set much further from threshold.

## [Benchmark results](@id ctrnn-results)

Every failed check of this backend in the benchmark suite, and what it missed. [Results](@ref) has every case.
This backend is run at noise 0.1 only, on all ten example networks.

```@eval
Main.ResultsTables.misses("ctrnn")
```

## API

[`CTRNNDesign`](@ref), [`CTRNNModel`](@ref), and in the internals [`ctrnn_layout`](@ref),
[`ctrnn_passage_time`](@ref), [`ctrnn_correct_offsets`](@ref) and [`ctrnn_parameters`](@ref).
