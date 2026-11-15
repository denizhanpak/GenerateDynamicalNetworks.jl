```@meta
CurrentModule = GenerateDynamicalNetworks
```

# The acceptance rule

A model is judged by simulating it on seeds the fit never used and comparing what it does with the
specification. The same rule is applied to every backend.

## One check

[`validate`](@ref) runs the model on 64 seeds, each for 250 times the sum of the dwells, and applies
[`verdict`](@ref). The check passes if all of these hold.

1. **Dwells.** Every state's mean dwell is within 5% of its target.
2. **Branches.** For every state with two or more successors, the measured share of each branch is inside the
   95% binomial interval around its designed probability:
   ```math
   |\hat p_{ij} - P_{ij}| \le 1.96\sqrt{\frac{P_{ij}(1 - P_{ij})}{n_i}} ,
   ```
   where ``n_i`` is the number of transitions out of state ``i``.
3. **Off-graph transitions.** Fewer than 1% of all transitions are between states the graph doesn't connect.
4. **No failed runs.** No run left the model's bounds.

```julia
check = validate(model, spec)
check.pass
check.dwell      # mean dwell per state
check.err        # relative dwell errors
check.branch     # one (; from, to, designed, measured, ok) per branch
check.off        # off-graph share
```

## The protocol

[`accept`](@ref) makes three checks, and a model is accepted if it passes all three.

| Check | Seeds | Step |
|:--|:--|:--|
| Set 1 | `9_000_001:9_000_064` | default |
| Set 2 | `9_200_001:9_200_064` | default |
| Half step | `9_000_001:9_000_064` | half the default |

The second set shows whether a pass depended on the particular runs. The half step shows whether it depended
on the step.

```julia
a = accept(model, spec)
a.pass
a.set1, a.set2, a.half_step     # the three checks
```

## Held-out seeds

The fits simulate too, to correct their parameters, and they draw their seeds from other ranges (from
100 000, 200 000, 1 000 000 and 8 000 000 upward). The acceptance seeds start at 9 000 001, so the model is
always judged on runs it wasn't fitted to.

## How strict the branch rule is

The interval of the branch rule narrows as the runs get longer. In a check on the `"foraging"` network the
branching state is left about 20 000 times, and the interval is 0.6 percentage points either side of the
designed probability. Two things follow.

- **A correct model can fail.** A model whose branch probabilities are exactly right falls outside a 95%
  interval about one time in twenty, for each branching state and each check.
- **A slightly wrong model can pass.** A model that is off by half a percentage point passes some checks and
  fails others. [A branching graph](@ref) shows one: it passes all three checks of the protocol, its share
  is about half a point too high on average, and it fails four of ten further sets of runs.

So one failed branch check, with the measured share within a percentage point or so of the design, doesn't
separate these two cases, and neither does a pass. Most failures of the stable backends in the benchmark suite
are of this kind: one check of the three missed, by about a percentage point, with every dwell well inside its
band ([How the cases fail](@ref)). A dwell outside its band, or a branch that misses on all three checks, is a
clearer sign of a model that is wrong.

Read the measured values, not only the verdict.

The rule was fixed before the benchmark runs and is kept as it is, so results stay comparable.

## The fitting objective

The fits don't use the verdict, which is pass or fail. They minimise [`score`](@ref), a smooth measure of the
same things: the squared log ratio of measured to target dwell for each state, the squared log ratio of
measured to designed share for each branch, and a penalty on off-graph transitions above 0.5%.
