```@meta
CurrentModule = GenerateDynamicalNetworks
```

# [Simplex](@id simplex-backend)

!!! warning "Status: experimental"
    Fails the acceptance protocol on half or more of the benchmark cases; see
    [Benchmark results](@ref simplex-results) below.

The simplex realisation of [AshwinPostlethwaite2013](@citet): one cell per state, with each state an
equilibrium on a coordinate axis. It is the same kind of system as the [Horchler](@ref horchler-backend)
backend builds, with two differences. Every edge has its own rate, and the branch probabilities come from those
rates, not from different noise on different nodes. So there is one noise level and there are no hidden states.

```julia
model = fit(spec, Simplex())
```

## [The system](@id simplex-system)

One variable ``u_j`` per state, with activity ``u_j^2``:

```math
du_j = s\,u_j\Big(1 - |u|^2 + \sum_i a_{ij}\,u_i^2\Big)\,dt + \varepsilon\,dW_j .
```

State ``i`` is the equilibrium with ``u_i = 1`` and every other variable zero. There, ``u_j`` grows at the rate
``s\,a_{ij}``, so the matrix ``a`` says how each state is left:

```math
a_{ij} =
\begin{cases}
e_{ij} > 0 & \text{if } i \to j \text{ is an edge} \\[2pt]
-c_i < 0 & \text{if it isn't} \\[2pt]
0 & i = j .
\end{cases}
```

The parameters are a global speed ``s``, an expansion rate ``e_{ij}`` per edge and a contraction ``c_i`` per
state. The noise ``\varepsilon`` is the specification's, the same on every variable.

The model is stored as an [`LVModel`](@ref) in the `u` form, the type the Horchler backend also builds: with a
shared rate, the Horchler design is this system with ``e = 1/\nu`` and ``c = 1``. A frame is read as the cell
with the largest ``|u_j|``.

## The design

### Branches from rates

Where a state has two successors, both grow from the noise level, and the one with the larger rate usually gets
there first. How strongly the rates decide depends on the noise: the lower it is, the longer the race and the
more a small difference in rate counts. The backend uses the low-noise scaling

```math
\frac{e_{ij}}{e_{ik}} \approx \Big(\frac{p_{ij}}{p_{ik}}\Big)^{1/L}, \qquad L = \max\!\big(\ln(\sqrt{0.1}/\varepsilon),\,1\big) .
```

At noise 0.001, ``L \approx 5.8``: a branch of 0.7 to 0.3 needs rates that differ by only about 16%.

### The starting point

[`edge_seed_rates`](@ref) takes the speed and a rate per state from the Horchler dwell formula with one noise
value ([Horchler](@ref horchler-backend), step 3), and splits each state's rate among its edges by the scaling
above. Contractions start at 1. If the dwell formula has no solution, a rough guess from the dwells is used.

## The fit

1. **Correction passes.** After each pass the edges out of a state move by the log of measured over target
   dwell, and the edges of a branch also by the branch error divided by ``L``. There are `passes` passes of
   `runs` runs.
2. **Held-out check.** The pass with the best score is checked on 32 further runs and returned if it passes.
3. **Search.** Otherwise a CMA-ES search over every parameter, for at most `budget` evaluations or `seconds`
   seconds.
4. **Polish.** Two more correction passes, of 32 runs four times as long.

| Option | Default | Meaning |
|:--|:--|:--|
| `passes` | 4 | correction passes |
| `runs` | 16 | runs per pass and per search evaluation |
| `T` | `125sum(spec.τ)` | length of each run |
| `budget` | 500 | most search evaluations |
| `popsize` | 12 | CMA-ES population |
| `seconds` | 300 | time limit of the search |

!!! warning "The time limit"
    If the search stops because of `seconds`, the result depends on how fast the machine is. Use `seconds=Inf`
    to limit the search by `budget` alone and get the same model everywhere.

## Limits

- **Branch probabilities are steep in the rates.** At low noise a small error in a rate ratio is a large error
  in the branch shares, and the correction passes measure the shares on few runs (16, then 32). The branch
  errors left over are larger than the acceptance rule allows in most benchmark cases.
- **Which graphs.** The realisation is for graphs with no state that follows itself and no two-way edge. The
  backend doesn't check this, and it hasn't been run on the two-way example networks.
- **The time limit**, as above.

## [Benchmark results](@id simplex-results)

Every failed check of this backend in the benchmark suite, and what it missed. [Results](@ref) has every case.

```@eval
Main.ResultsTables.misses("simplex")
```

Half of the failing cases miss on all three checks, with the measured share on the same side each time: these
are consistent branch errors, not chance. One case misses a dwell, on one check only.

The models built for these networks are all classified as heteroclinic.

## API

[`Simplex`](@ref), [`LVModel`](@ref), and in the internals [`simplex_model`](@ref), [`edge_seed_rates`](@ref),
[`correct_edges`](@ref) and [`edge_rate_fit`](@ref).
