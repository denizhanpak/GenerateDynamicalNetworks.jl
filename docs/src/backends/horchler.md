```@meta
CurrentModule = GenerateDynamicalNetworks
```

# [Horchler](@id horchler-backend)

!!! note "Status: stable"
    Passes the acceptance protocol on most of the benchmark cases; see [Benchmark results](@ref horchler-results)
    below.

A Lotka–Volterra network with one node per state. Each state is a saddle point at which one node is active, and
the system moves along the connections between saddles in the order the graph gives. The rates are designed in
closed form from the dwell times, following [Horchler2015](@citet), and then corrected by simulation.

```julia
model = fit(spec, Horchler())      # the u form, the default
model = fit(spec, Horchler(:x))    # the x form
```

## [The system](@id horchler-system)

The model ([`LVModel`](@ref)) has an activity ``x_i \ge 0`` for each node. In the `:x` form

```math
dx_i = x_i\Big(\alpha_i - \sum_j \rho_{ij}\,x_j\Big)\,dt + \varepsilon_i\,dW_i ,
```

with ``x`` put back to ``x \ge 0`` after every step. In the `:u` form the variable is ``u_i`` with ``x_i = u_i^2``:

```math
du_i = u_i\Big(\alpha_i - \sum_j \rho_{ij}\,u_j^2\Big)\,dt + \varepsilon_i\,dW_i .
```

Without noise the two are the same system, with every rate doubled in the `u` form. They differ in how the
noise enters, and the `x` form can be designed only at much lower noise than the `u` form.

A frame is read as the node with the largest ``|u_i|``.

### Saddles

Node ``j`` alone at activity ``\beta_j = \alpha_j / \rho_{jj}`` is an equilibrium. The package uses
``\beta_j = 1`` throughout. The coupling matrix is set so that this equilibrium is a saddle that can be left
only towards the successors of state ``j``:

```math
\rho_{ij} =
\begin{cases}
\alpha_i / \beta_i & i = j \\[2pt]
(\alpha_i - \alpha_j / \nu_j) / \beta_j & \text{if } j \to i \text{ is an edge} \\[2pt]
(\alpha_i + \alpha_j) / \beta_j & \text{otherwise.}
\end{cases}
```

At node ``j`` the activity of another node ``i`` grows at the rate ``\alpha_i - \rho_{ij}\beta_j``. That is
``\alpha_j / \nu_j`` if ``i`` follows ``j`` and ``-\alpha_j`` if it doesn't. So each state has

- a contraction rate ``\alpha_j``, and
- an expansion rate ``\alpha_j / \nu_j`` towards each successor.

``\nu_j`` is the *saddle value*, the ratio of contraction to expansion. The larger it is, the more slowly the
system leaves state ``j``.

## The design

[`horchler_design`](@ref) turns `(P, τ, noise)` into ``\alpha``, ``\nu``, ``\varepsilon`` and ``\rho`` in four
steps.

### 1. Branch probabilities from noise

Where a state has two successors ``j`` and ``k``, the one whose activity is larger when the system leaves wins.
Both grow at the same rate, so the odds depend only on their noise amplitudes:

```math
\frac{\varepsilon_j}{\varepsilon_k} = \tan\!\Big(\frac{\pi\,p_j}{2}\Big) ,
```

where ``p_j`` is the probability of going to ``j``. Each two-way branch gives one such equation in the logs of
the noise amplitudes. The design takes the solution closest to equal noise on every node and multiplies it by
the specification's noise, so a node that isn't the target of a branch has exactly the specification's noise.

A branch with three or more successors is supported only when they are equally likely, and then they get equal
noise.

### 2. Effective noise per state

A state with several exits leaves sooner than one with a single exit at the same noise, because the exits race.
[`effective_noise`](@ref) gives the single-exit noise ``\varepsilon^{\text{eff}}_j`` with the same mean escape
time.

### 3. Dwell times

The designed mean dwell of a state is the mean passage time past its saddle [StoneHolmes1990](@cite) plus the
time to travel to the next one:

```math
\tau_j = \frac{1}{2\lambda_u}\Big[\,2\ln\frac{\delta}{\varepsilon^{\text{eff}}_j} + \ln 4\lambda_u
         - \ln\!\Big(1 + \frac{\lambda_u}{\lambda_s}\Big) + \gamma\Big] + \frac{c\,\ln(1/\kappa - 1)}{\alpha_j},
\qquad \lambda_u = \frac{\alpha_j}{\nu_j},\quad \lambda_s = \alpha_j .
```

Here ``\gamma`` is Euler's constant and ``\kappa = 0.1`` sets the size of the neighbourhood that counts as
being at a saddle. In the `u` form ``\delta = \sqrt{\kappa\beta_j}`` and ``c = 1``; in the `x` form
``\delta = \kappa\beta_j`` and ``c = 2``.

[`dwell_rates`](@ref) inverts this.

- One rate ``\alpha`` is shared by every state. It is the largest any state needs when its saddle value is at
  its minimum, ``\nu_{\min} = 2``.
- Each state's ``\nu_j`` is then found by root finding so that its designed dwell is ``\tau_j``.

At a given ``\alpha`` and noise there is a longest dwell the formula can give. A request beyond it throws an
`ArgumentError` saying which state and how long a dwell the noise allows.

```@example horchler
using GenerateDynamicalNetworks
const GDN = GenerateDynamicalNetworks

d = GDN.horchler_design(ring(3), [1.0, 3.0, 10.0]; noise=0.1)
d.α, d.ν
```

Here the state with the shortest dwell sets ``\alpha`` and sits at the smallest saddle value; the longer the dwell, the larger the saddle value.

### 4. The coupling matrix

``\rho`` follows from ``\alpha``, ``\nu`` and the graph as in [The system](@ref horchler-system).

## Graphs that need hidden states

The construction can't represent three kinds of graph directly, so [`repair`](@ref) first changes the graph
by putting a hidden state on an edge:

| Problem | Example | Hidden state goes on |
|:--|:--|:--|
| a two-way edge | 1 ⇄ 2 | one of the two directions |
| a Δ-clique: a state reachable both directly and through another | 1 → 2 → 3 and 1 → 3 | the edge 2 → 3 |
| two branching states whose branches need different noise on the same node | `"shared, conflicting"` | one of the shared edges |

The first is the condition of the simplex realisation of [AshwinPostlethwaite2013](@citet), which needs a
graph with no state that follows itself and no two-way edge. The third comes from setting branch probabilities
by noise: each node has one noise amplitude, and two branches that share a successor may ask for two different
ones.

A hidden state on the edge ``a \to b`` takes half of ``a``'s dwell, and ``a`` gives up as much on average, so
the mean time from entering ``a`` to entering the next visible state is unchanged. The decoder doesn't report
hidden states: while one is the most active node, the last visible state holds.

```@example horchler
r = GDN.repair([0 1; 1 0], [1.0, 0.3])    # the flip-flop
r.reasons, r.P, r.τ
```

A graph that still can't be used raises an `ArgumentError`. That happens for a state that follows itself and
for a branch of three or more successors with unequal probabilities.

## The fit

The closed form is a leading-order approximation, so `fit` corrects the designed model by simulation.

1. **Correction passes.** The design is made for *requested* dwells, which start at the targets. After each
   pass, every request is multiplied by target over measured dwell, and the design is made again. There are
   `passes` passes of `runs` runs each, on new seeds every pass.
2. **Held-out check.** The pass with the best score ([The fitting objective](@ref)) is checked on 32 further
   runs. If it passes, it is returned.
3. **Search.** Otherwise a CMA-ES search over every node's ``\alpha`` and ``\nu`` starts from the first design.
   It stops after `budget` evaluations or `seconds` seconds.
4. **Polish.** Two passes then scale each visible state's ``\nu`` by target over measured dwell.

| Option | Default | Meaning |
|:--|:--|:--|
| `form` | `:u` | the `:u` or `:x` form |
| `passes` | 4 | correction passes |
| `runs` | 16 | runs per pass and per search evaluation |
| `T` | `125sum(spec.τ)` | length of each run |
| `budget` | 500 | most search evaluations |
| `popsize` | 12 | CMA-ES population |
| `seconds` | 300 | time limit of the search |

!!! warning "The time limit"
    If the search stops because of `seconds`, the result depends on how fast the machine is. Use `seconds=Inf`
    to limit the search by `budget` alone and get the same model everywhere.

`fit(spec, Horchler(); verbose=true)` prints whether the search ran and how many evaluations it used.

## Limits

- **Dwells too long for the noise.** The longer the dwell and the larger the noise, the larger the saddle value
  needed, and beyond a point no value gives the dwell. The fit then throws an `ArgumentError`. The `x` form
  reaches this limit at much lower noise than the `u` form.
- **Branches are set by formula only.** The noise ratios of step 1 are never adjusted: the correction passes
  change the dwells, and the search changes ``\alpha`` and ``\nu``. A branch probability that the formula gets
  slightly wrong stays slightly wrong.
- **Hidden states change the shape of a dwell.** A visit that passes through a hidden state is two passages
  in a row. Its mean is as requested, but its distribution isn't that of a single passage.
- **Uneven branches of three or more** and **states that follow themselves** aren't supported.

## [Benchmark results](@id horchler-results)

Every failed check of this backend in the benchmark suite, and what it missed. No dwell was outside its band;
all the failures are branch checks. [Results](@ref) has every case, including the ones that passed.

```@eval
Main.ResultsTables.misses("horchler")
```

The models built for these networks are all classified as heteroclinic.

## API

[`Horchler`](@ref), [`LVModel`](@ref), and in the internals [`horchler_design`](@ref), [`repair`](@ref),
[`dwell_rates`](@ref), [`stone_holmes_mean`](@ref) and [`effective_noise`](@ref).
