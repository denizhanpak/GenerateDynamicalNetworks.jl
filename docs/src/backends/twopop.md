```@meta
CurrentModule = GenerateDynamicalNetworks
```

# [Two-population](@id twopop-backend)

!!! warning "Status: experimental"
    Fails the acceptance protocol on half or more of the benchmark cases; see
    [Benchmark results](@ref twopop-results) below.

The two-population network of [AshwinPostlethwaite2016](@citet): one population of cells for the states and
one for the edges. Each edge has a parameter whose sign decides whether the state is left as a saddle or over a
threshold, so a fitted model can be heteroclinic, excitable or a mixture. One noise level, no hidden states.

```julia
model = fit(spec, TwoPopulation())
```

## [The system](@id twopop-system)

A cell ``p_j`` per state and a cell ``y_k`` per edge. Edge ``k`` goes from state ``\alpha_k`` to state
``\omega_k``.

```math
\begin{aligned}
dp_j &= s\Big[p_j\big(F(1 - |p|^2) + D(p_j^2|p|^2 - |p|^4)\big) + E\,(Z^{\text{in}}_j - Z^{\text{out}}_j)\Big]dt + \varepsilon\,dW , \\
dy_k &= -s\,y_k\Big[(y_k^2 - 1)^2 + A - B_k\,p_{\alpha_k}^2 + C\big(|y|^2 - y_k^2\big)\Big]dt + \varepsilon\,dW ,
\end{aligned}
```

```math
Z^{\text{out}}_j = \sum_{k:\,\alpha_k = j} y_k^2\,p_j\,p_{\omega_k}, \qquad
Z^{\text{in}}_j = \sum_{k:\,\omega_k = j} y_k^2\,p_{\alpha_k}^2 ,
```

with the paper's constants ``A = 0.5``, ``C = 2``, ``D = 10``, ``E = 4``, ``F = 2``, and ``B_k = 1.5 - \nu_k``.

State ``j`` is the point with ``p_j = 1`` and everything else zero. While the system is there, the cell of an
edge ``k`` leaving ``j`` grows at the rate ``-s\,\nu_k``:

- ``\nu_k < 0``: the edge cell is unstable at the state. Noise sets it off and it carries the system to the
  next state. This is the heteroclinic side.
- ``\nu_k > 0``: the edge cell is stable at the state, and noise has to push it over a threshold. This is the
  excitable side.

The parameters are the global speed ``s`` and one ``\nu_k`` per edge. A frame is read as the state cell with
the largest ``|p_j|``.

## The design

The starting point is [`edge_seed_rates`](@ref), as in the [Simplex](@ref simplex-backend) backend: a speed
and a rate per state from the Horchler dwell formula, split among each state's edges by the low-noise scaling
of branch probabilities. Every edge starts on the heteroclinic side, with ``-\nu_k`` in proportion to its rate.

## The fit

1. **Correction passes** on the heteroclinic side: the edges out of a state move by the log of measured over
   target dwell, and the edges of a branch also by the branch error.
2. **Held-out check** on 32 further runs. If the best pass passes, it is returned.
3. **Search.** Otherwise a CMA-ES search over the speed and the signed ``\nu_k``, between −5 and 0.3, so an
   edge can cross to the excitable side. It stops after `budget` evaluations or `seconds` seconds.
4. **Polish.** If every ``\nu_k`` is still negative, two more correction passes.

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

- **Branch errors.** Every failure in the benchmark suite is a branch check, and no dwell is outside its band.
  Most failing cases miss on two or three checks in the same direction, by about one percentage point.
- **No polish after crossing over.** The correction passes work on the heteroclinic side only. A search result
  with any edge on the excitable side is returned as the search left it.
- **Two-way edges** haven't been tested: the backend was run on the one-way example networks only.
- **The time limit**, as above.

## [Benchmark results](@id twopop-results)

Every failed check of this backend in the benchmark suite, and what it missed. [Results](@ref) has every case.

```@eval
Main.ResultsTables.misses("twopop")
```

The models built for these networks are classified as heteroclinic, except the worm ring at noise 0.1, whose
class is unknown.

## API

[`TwoPopulation`](@ref), [`TwoPopModel`](@ref), and in the internals [`edge_seed_rates`](@ref) and
[`correct_edges`](@ref).
