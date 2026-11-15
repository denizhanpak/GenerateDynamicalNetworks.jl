```@meta
CurrentModule = GenerateDynamicalNetworks
```

# The specification

A [`TransitionSpec`](@ref) holds what a model has to reproduce. It has four parts.

| Field | Meaning |
|:--|:--|
| `P` | the transition matrix: `P[i, j]` is the probability that a visit to state `i` is followed by a visit to `j` |
| `τ` | the mean dwell: `τ[i]` is the mean length of a visit to state `i` |
| `noise` | the noise level ε of the stochastic system |
| `sample` | the frame interval: how often the decoder reads which state the system is in |

```@example spec
using GenerateDynamicalNetworks

P = [0 1 0 0; 0 0 0.7 0.3; 0 0 0 1; 1 0 0 0]
spec = TransitionSpec(P, [2.0, 1.0, 0.5, 1.5]; noise=0.1)
```

The rows of `P` sum to 1. The frame interval is a tenth of the shortest dwell unless you give `sample`. States
can be named with `names`.

A graph alone is enough when every branch out of a state is equally likely:

```@example spec
using Graphs
TransitionSpec(cycle_digraph(3), [1.0, 3.0, 10.0]; noise=0.1)
```

## What is checked

The constructor throws an `ArgumentError` listing everything wrong with the request. [`spec_problems`](@ref)
returns the same list without throwing:

```@example spec
spec_problems([0 1 0; 0 0 1; 0.5 0 0], [1.0, -2.0, 1.0], 0.0)
```

Two of the rules are less obvious than the rest.

- **The noise must be positive.** Most of these systems switch states only because noise pushes them off a
  saddle or over a threshold, and the weaker the noise the longer the wait. Without noise the dwell times
  aren't set.
- **Every state must be reachable from every other.** Otherwise a run ends up in part of the graph and the
  remaining states have no dwell to measure.

A backend can add its own limits. The Horchler backend, for example, can't use a graph with a state that
follows itself; see each backend's page.

## Noise and the frame interval

Both are part of the specification because the dwell times depend on them.

The noise sets the time scale of every escape, so a model fitted at one noise level has other dwells at
another. A model is built for the noise level in its specification and carries that level with it.

The frame interval is how the continuous state becomes a sequence of visits: the decoder reads the state once
per frame, as a camera would ([Simulation and decoding](@ref)). A visit shorter than a frame can be missed, and
the measured dwells are multiples of the frame, so the same model read at a different interval has slightly
different statistics.

## Example networks

Ten small networks come with the package. [`example`](@ref) returns one as a specification.

```@example spec
example("foraging"; noise=0.1)
```

| Name | States | What it has |
|:--|:--|:--|
| `"foraging"` | 5 | one branch (0.7 and 0.3) that rejoins |
| `"worm ring"` | 4 | a ring with very uneven dwells (5.3 to 0.21) |
| `"figure-eight"` | 5 | two loops through one state |
| `"Δ-clique"` | 4 | a state reachable both directly and through another (1 → 2 → 3 and 1 → 3) |
| `"shared, solvable"` | 6 | two branching states with a successor in common |
| `"shared, conflicting"` | 6 | two branching states choosing between the same two successors at different odds |
| `"flip-flop"` | 2 | two states, each following the other |
| `"two-way and exit"` | 3 | a two-way edge with a way out |
| `"hub with two returns"` | 3 | one state with two-way edges to two others |
| `"worm with return"` | 4 | the worm ring with a return from state 2 to state 1 |

The first six have one-way edges only. The last four have two-way edges, which some constructions can't
represent directly.

The dwells of the two worm networks are the mean bout lengths of the four locomotion states of *C. elegans*
printed in [Roberts2016](@citet), Table 1, and their frame interval is that study's camera rate (29.97 Hz).
The other networks are made up.

[`ring`](@ref) gives the transition matrix of a cycle of any length, and [`random_network`](@ref) draws a
random strongly connected graph with branches of at most two.

## Reading the graph

Backends number their per-edge parameters in the order [`edge_list`](@ref) gives, and [`successors`](@ref)
lists the states that can follow one:

```@example spec
edge_list(spec)
```

```@example spec
successors(spec, 2)
```
