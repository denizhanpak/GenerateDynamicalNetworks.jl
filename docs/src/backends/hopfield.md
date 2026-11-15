```@meta
CurrentModule = GenerateDynamicalNetworks
```

# [Hopfield](@id hopfield-backend)

!!! note "Status: stable"
    Passes the acceptance protocol on most of the benchmark cases; see
    [Benchmark results](@ref hopfield-results) below.

A Hopfield network whose neurons each sit near one of two values, so that the network's states are the
vertices of a cube. The graph is laid out on those vertices as in the [Synchrony](@ref synchrony-backend)
backend, but the coupling is the ordinary pairwise coupling of a Hopfield network: a matrix ``J`` and a bias
``b``.

```julia
model = fit(spec, Hopfield())
```

## [The system](@id hopfield-system)

[ChossatKrupa2016](@citet) write the Hopfield firing-rate equation on the cube ``[-1, 1]^n`` as

```math
\dot x_j = (1 - x_j^2)\Big(\lambda\,(Jx)_j + \lambda\,b_j - f(x_j)\Big) .
```

Every face of the cube is invariant, so the ``2^n`` vertices are equilibria. With ``f(x) = x`` and
``x_j = \cos\psi_j``, and ``\lambda`` absorbed into ``J`` and ``b``, each neuron obeys

```math
d\psi_j = \sin\psi_j\,\big(g_j + k_j\cos\psi_j\big)\,dt + \varepsilon\,dW_j, \qquad
g_j = -\Big(b_j + \sum_{m \ne j} J_{jm}\cos\psi_m\Big) .
```

This is the synchrony backend's equation for one pair, with coupling between pairs of neurons only. Noise acts
on the angle ``\psi_j``, so ``x_j`` stays in ``[-1, 1]``.

At a vertex, where every ``x_m = \pm 1``, neuron ``j`` moves away from its value at the rate

```math
r_j = k_j - x_j\Big(b_j + \sum_{m \ne j} J_{jm}\,x_m\Big) ,
```

which is linear in neuron ``j``'s own parameters ``(k_j, b_j, J_{j\cdot})``. So each neuron can be designed
separately. A frame is read from the signs of ``\cos\psi_j``, which give the vertex.

### What pairwise coupling costs

In the synchrony backend every rate at every corner is free. Here a neuron has one row of parameters for all
vertices, and two things follow.

- **Leaving and arriving are tied.** The two ends of a cube edge along neuron ``j`` differ only in ``x_j``, so
  their rates add up to ``2k_j``. An edge left at rate ``r`` is arrived at with rate ``2k_j - r``. The design
  sets ``k_j`` from the slowest departure along ``j`` so that every arrival contracts at least at the
  contraction rate; faster departures along the same neuron arrive harder.
- **Not every layout has a design.** The vertices at which a neuron flips must be separable from the others by
  a threshold on the remaining neurons. A layout whose flips aren't of that kind can't be built at all.

## The design

### The layout

[`hop_layout`](@ref) first searches as the synchrony backend does, for the fewest neurons, and keeps only
layouts whose every neuron can be designed at the rule-of-thumb rates. The search stops at seven neurons:
random layouts of a larger cube are less likely to have a design, not more.

If the search finds nothing, a structured layout is used: one neuron per state, with state ``i`` the vertex
where only neuron ``i`` is on. An edge ``i \to j`` switches ``j`` on and then ``i`` off. Each two-way pair gets
a relay neuron for one of its directions.

### One neuron at a time

[`hop_design`](@ref) solves a small quadratic program for each neuron (JuMP with OSQP).

- Every departure along the neuron has exactly its rate.
- At every other vertex in use, the neuron contracts, by at least a quarter of the contraction rate and aiming
  at the full rate.
- Unused vertices next to the network point back towards it, as far as possible.

### The starting rates

Departure rates start from the race calculation described for the synchrony backend
([The race seed](@ref)), here with each edge's own arrival rate, since those differ from edge to edge.

## The fit

The correction passes are the synchrony backend's ([`correct_rates`](@ref)): 32 runs per pass, 96 at half the
step in the last three, one Broyden step per state on its departure rates. Every rate is capped at twice the
contraction rate.

If the corrected rates of some pass have no design (a neuron's program can't be met), the fit warns and
returns the model of the last pass that had one.

| Option | Default | Meaning |
|:--|:--|:--|
| `passes` | 12 | correction passes |
| `base` | 100 000 | first seed; different values give independent fits |

A fit can be run in batches with `checkpoint` and `deadline`, as for the [CTRNN design](@ref ctrnn-backend).

## Limits

- **Rates the pairwise design can't hold.** `"shared, solvable"` at noise 0.1 is the one benchmark case with a
  dwell outside its band: state 6 comes out about 22% long on all three checks. In the research prototype
  the same case ended that way because the corrections drove the rates to where one neuron's program had no
  solution, and the fit kept the last pass that could be built.
- **Layouts are checked at the starting rates only.** A layout accepted at the rule-of-thumb rates can stop
  having a design as the rates are corrected, as above.
- **Branch errors.** The other failures in the benchmark suite are branch checks, within about a percentage
  point.
- **Only mean dwells are fitted.**
- **Two-way edges** are laid out, with relay neurons where needed, but the backend was run on the one-way
  example networks only.

## [Benchmark results](@id hopfield-results)

Every failed check of this backend in the benchmark suite, and what it missed. [Results](@ref) has every case.

```@eval
Main.ResultsTables.misses("hopfield")
```

The models built for these networks are classified as heteroclinic, except the worm ring at noise 0.1, whose
class is unknown.

## API

[`Hopfield`](@ref), [`HopModel`](@ref), and in the internals [`hop_layout`](@ref), [`hop_design`](@ref),
[`hop_seed`](@ref) and [`correct_rates`](@ref).
