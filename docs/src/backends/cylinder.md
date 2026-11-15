```@meta
CurrentModule = GenerateDynamicalNetworks
```

# [Cylinder](@id cylinder-backend)

!!! warning "Status: experimental"
    Fails the acceptance protocol on half or more of the benchmark cases; see
    [Benchmark results](@ref cylinder-results) below.

The cylinder realisation of [AshwinPostlethwaite2013](@citet). The states sit at the integers on a line, and
there is one cell per edge: an edge's cell switches on to carry the system from one state to the next. One
noise level, no hidden states.

```julia
model = fit(spec, Cylinder())
```

## [The system](@id cylinder-system)

A position ``p`` and one cell ``y_m`` per edge. Edge ``m`` goes from state ``\alpha_m`` to state ``\omega_m``,
and state ``j`` is the point ``p = j`` with every ``y_m = 0``.

```math
\begin{aligned}
dy_m &= -s\,y_m\,G_m(y, p)\,dt + \varepsilon\,dW_m , \\
dp &= s\Big(-\sin 2\pi p + \sum_m y_m^2 \tanh(\omega_m - p)\Big)\,dt + \varepsilon\,dW_0 ,
\end{aligned}
```

```math
G_m = \big(y_m^2 - \tfrac54\big)^2 - 1 - L_{\alpha m}\,\mathrm{sech}^2\big(k(p - \alpha_m)\big)
      + L_\omega\,\mathrm{sech}^2\big(k(p - \omega_m)\big) + \sum_{i \ne m} y_i^2 .
```

Without an active edge cell, ``p`` rests at an integer. Near the start of edge ``m`` its cell can grow; an
active cell pulls ``p`` towards the edge's end, where the cell is switched off again.

At state ``i`` the cell of an edge ``i \to j`` grows at the rate

```math
s\,e_{ij}, \qquad e_{ij} = L_{\alpha m} - \tfrac{9}{16} - L_\omega\,\mathrm{sech}^2\big(k(i - j)\big) ,
```

and the parameters of the model are the global speed ``s`` and one rate ``e_{ij}`` per edge. The steepness
``k`` and ``L_\omega`` are fixed options.

### The decoder

A frame is read as state ``j`` only when ``p`` is within `pwindow` of ``j`` and every ``|y_m|`` is below
`ywindow`. Otherwise the last state holds. This matters because the states are on a line: on the way from
state 4 to state 1, ``p`` passes 3 and 2, and those must not be read as visits.

## The design

As in the [Simplex](@ref simplex-backend) backend, the branch probabilities come from the edge rates, with the
same low-noise scaling, and the starting point is [`edge_seed_rates`](@ref): the speed and a rate per state
from the Horchler dwell formula, split among each state's edges. The speed and rates are then rescaled by the
contraction at a state, ``L_\omega + 9/16``.

## The fit

The same four steps as the simplex backend ([`edge_rate_fit`](@ref)): correction passes, a held-out check, a
CMA-ES search over the speed and the edge rates if the check fails, and two more correction passes.

| Option | Default | Meaning |
|:--|:--|:--|
| `k` | 2 | steepness of the switching functions |
| `Lω` | 1.5625 | ``L_\omega`` |
| `pwindow` | 0.25 | how close ``p`` must be to an integer to be read as a state |
| `ywindow` | 0.5 | how small every edge cell must be |
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

- **The worm ring at noise 0.1 fails outright.** State 1 lasts about half as long as it should, and a quarter
  of the transitions are between states the graph doesn't connect. The cause isn't known.
- **Branch errors.** The other six failing cases are branch errors of one to four percentage points. Four of
  them miss on all three checks, in the same direction each time.
- **Off-graph transitions at high noise.** In `"shared, solvable"` at noise 0.1 the two branches of state 4
  add up to about 0.95: the remaining transitions out of state 4 go to states it has no edge to.
- **The steepness.** The default `k = 2` was chosen in the research prototype. With the value printed for the
  paper's own example, the prototype's reading of the equations gave edges that couldn't leave their state.
- **Two-way edges.** The realisation needs only that no state follows itself, so it should take two-way edges
  without hidden states. That hasn't been tested: the backend was run on the one-way example networks only.
- **The time limit**, as above.

## [Benchmark results](@id cylinder-results)

Every failed check of this backend in the benchmark suite, and what it missed. [Results](@ref) has every case.

```@eval
Main.ResultsTables.misses("cylinder")
```

The models built for these networks are all classified as heteroclinic.

## API

[`Cylinder`](@ref), [`CylinderModel`](@ref), and in the internals [`edge_seed_rates`](@ref),
[`correct_edges`](@ref) and [`edge_rate_fit`](@ref).
