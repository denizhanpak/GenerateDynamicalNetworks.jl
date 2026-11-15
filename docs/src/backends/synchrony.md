```@meta
CurrentModule = GenerateDynamicalNetworks
```

# [Synchrony](@id synchrony-backend)

!!! note "Status: stable with the default seed, experimental with `seed=:race`"
    With the default seed it passes the acceptance protocol on most of the benchmark cases; with the race seed
    it fails half of them. See [Benchmark results](@ref synchrony-results) below.

Phase oscillators in pairs. Each pair is either in phase or in anti-phase, and a state of the graph is one
pattern of in-phase and anti-phase pairs. The network of states lives in *synchrony space*: every oscillator
keeps the same frequency throughout, and what changes is which pairs are synchronised.

```julia
model = fit(spec, Synchrony())              # the default seed
model = fit(spec, Synchrony(; seed=:race))  # the race seed
```

## [The system](@id synchrony-system)

``2M`` oscillators with one natural frequency, in ``M`` pairs. In the frame rotating at that frequency, each
oscillator of pair ``\sigma`` obeys

```math
d\theta_{\sigma,k} = \big(a_\sigma \sin\Delta_k + b_\sigma \sin 2\Delta_k\big)\,dt + \varepsilon\,dW ,
\qquad \Delta_k = \theta_{\sigma,\text{other}} - \theta_{\sigma,k} ,
```

where ``a_\sigma`` and ``b_\sigma`` depend on the other pairs through the cosines of their phase differences.
The coupling therefore involves more than two oscillators at a time, in the spirit of [Bick2018](@citet) and
[BickLohse2019](@citet). The phase difference ``\psi_\sigma`` of a pair then obeys

```math
\dot\psi_\sigma = \sin\psi_\sigma\,\big(g_\sigma + k_\sigma\cos\psi_\sigma\big), \qquad g_\sigma = -2a_\sigma,\quad k_\sigma = -4b_\sigma .
```

Two things follow from the factor ``\sin\psi_\sigma``.

- In phase (S, ``\psi_\sigma = 0``) and anti-phase (D, ``\psi_\sigma = \pi``) are invariant for every pair,
  whatever the other pairs do.
- The ``2^M`` patterns of S and D are equilibria. They are the corners of an ``M``-dimensional cube, and moving
  along a cube edge flips one pair.

At a corner, the rate at which pair ``\sigma`` moves away from its current value is
``\lambda_\sigma = k_\sigma + g_\sigma`` if it is in phase and ``k_\sigma - g_\sigma`` if it is in anti-phase.
Because ``g_\sigma`` and ``k_\sigma`` can be chosen separately for every pattern of the other pairs, **every
rate at every corner is free**. A cube edge that is left at a positive rate and arrived at with a negative one
carries a connection from one corner to the other.

A frame is read from the signs of ``\cos\psi_\sigma``, which give the corner.

## The design

### The layout

[`sync_layout`](@ref) places the graph on the cube.

- Each state is a corner.
- Each edge is a path of single flips from its state's corner to the next state's. Corners along the way are
  hidden *transit* corners, and time spent there counts towards the state being left.
- Edges out of one state start along different pairs, and no corner is used by two paths. A two-way edge needs
  two separate paths.

The search tries random placements, for the fewest pairs that work and then the fewest transit corners.

```@example synchrony
using GenerateDynamicalNetworks
const GDN = GenerateDynamicalNetworks

spec = example("foraging"; noise=0.1)
layout = GDN.sync_layout(spec.P)
layout.M, [GDN.corner_name(v, layout.M) for v in layout.corner_of], GDN.transits(layout)
```

Five states on a cube of three pairs (six oscillators), with two transit corners.

### The rates

[`corner_rates`](@ref) fills in every rate from one departure rate per edge.

| Where | Rate |
|:--|:--|
| leaving a state along an edge | that edge's departure rate, which the fit adjusts |
| leaving a transit corner | a fast rate |
| every other direction at a used corner | contracting, at the same fast rate |
| unused corners | pointing back towards the network |

The fast rate is four times the largest rule-of-thumb departure rate.

### The starting rates

With the default `seed=:rough`, each departure rate starts from a rule of thumb: the time to leave a saddle
at noise ``\varepsilon`` is about ``\ln(1/\varepsilon)/\lambda``, so

```math
\lambda_{ij} = \frac{\ln(1/\varepsilon)}{\tau_i}\,\sqrt{P_{ij}\cdot(\text{number of successors of } i)} .
```

## The fit

[`correct_rates`](@ref) corrects the departure rates from simulation.

- Each pass simulates the model on new seeds: 32 runs of 250 times the sum of the dwells. The last three
  passes use 96 runs at half the step and take smaller steps.
- For each state, the pass compares the measured dwell and branch ratios with their targets and takes one
  Broyden step on the departure rates of the state's edges. How the measurements respond to the rates is
  learned from pass to pass.

A departure rate may be corrected to below zero. The corner is then weakly stable, and noise has to push the
pair over a small barrier: that state has moved to the excitable side. This happens for long dwells at high
noise, where even a rate of zero leaves the corner too soon.

| Option | Default | Meaning |
|:--|:--|:--|
| `seed` | `:rough` | the starting rates: `:rough` or `:race` |
| `passes` | 12 | correction passes |
| `base` | 100 000 | first seed; different values give independent fits |

A fit can be run in batches with `checkpoint` and `deadline`, as for the [CTRNN design](@ref ctrnn-backend).

## The race seed

!!! warning "Status: experimental"
    With `seed=:race` the backend fails half of the benchmark cases.

The rule of thumb treats each exit on its own. `seed=:race` starts instead from a calculation of the whole
race out of a state. Each exit is close to a one-dimensional diffusion of one phase difference, whose waiting
time distribution can be computed on a grid; a state's exits race each other, which gives the dwell and the
branch shares that any set of rates should produce ([`race`](@ref)). The seed solves for the rates that give
the target dwell and shares, and the correction passes start from there, with their first response taken from
the same calculation.

In the benchmark suite the race seed passes fewer cases than the default ([Results](@ref)). All of its
failures are a single branch check on one set of runs, each within about a percentage point.

In the research prototype the race seed made the difference on one graph that isn't in the suite: one with
branches of about 1% out of slow states, which failed with the default seed and nearly passed with the race
seed (`research/notebooks/phase6_synchrony.jl`).

## Limits

- **Size.** The layout search stops at seven pairs (14 oscillators) and throws an `ArgumentError` if no
  layout is found. The cube doubles with every pair.
- **Branch errors.** Every failure in the benchmark suite is a branch check; every dwell is inside its band.
  With the default seed, two of the four failing cases miss on two checks in the same direction.
- **Rare branches from slow states** are where the default seed struggled in the prototype, as above.
- **Only mean dwells are fitted.**
- **Two-way edges** are laid out, but the backend was run on the one-way example networks only.

## [Benchmark results](@id synchrony-results)

Every failed check with the default seed, and what it missed. [Results](@ref) has every case.

```@eval
Main.ResultsTables.misses("synchrony")
```

With the race seed:

```@eval
Main.ResultsTables.misses("synchrony_race")
```

Most of the models are classified as heteroclinic. The rings with long dwells at noise 0.1 come out as
unknown. On the three-state ring with dwells 1, 3 and 10, the fitted departure rate of the longest state is
slightly negative and the other two are positive, so one corner is on the excitable side and two are saddles.
The classifier's noise-free runs then end at the stable corner from some starting points and at a saddle from
others, and starting points that disagree give the class unknown.

## API

[`Synchrony`](@ref), [`SyncModel`](@ref), and in the internals [`sync_layout`](@ref), [`corner_rates`](@ref),
[`rough_rates`](@ref), [`correct_rates`](@ref), [`race`](@ref) and [`race_seed_rates`](@ref).
