```@meta
CurrentModule = GenerateDynamicalNetworks
```

# Simulation and decoding

Every backend builds a stochastic differential equation

```math
du = f(u)\,dt + g(u)\,dW ,
```

and one simulator runs all of them. [`simulate`](@ref) returns the sequence of states each run visits, not the
trajectory `u(t)`.

```julia
runs = simulate(model, spec; T=500, seeds=1:16)
runs[1].state     # the states entered, in order
runs[1].t         # when each was entered
runs[1].failed    # whether the run left the model's bounds
```

## The step

The equation is integrated by the Euler–Maruyama method at a fixed step. Each model reports its fastest rate,
and the step is the largest one that is no coarser than `dtscale` divided by that rate and that lands exactly
on every frame. The default `dtscale` is 0.01, so the default step is at most a hundredth of the model's
fastest time scale.

The acceptance protocol repeats one of its checks at half the step, to catch a model whose statistics depend
on the step ([The acceptance rule](@ref)).

## Frames and the decoder

The state is read once per frame, at the interval `sample` of the specification. Each model has a decoder
that turns the variables `u` at that instant into a state number. The Horchler model, for example, reads the
node with the largest activity.

A decoder can also answer "no state": the system is between states, or at a hidden state the specification
doesn't have. The last state then holds. A visit is recorded only when the decoded state changes, so a run is
a list of visits, each with the time it began.

Two things follow from reading by frames.

- A visit lasts a whole number of frames.
- An excursion to another state that begins and ends inside one frame isn't seen.

## Runs and seeds

`simulate` makes one run per seed, in parallel over threads. A run depends only on its seed, not on the number
of threads or on the other runs, so results can be reproduced exactly on the same version of the package and
its dependencies.

A run that leaves the model's bounds is stopped and marked `failed`. By default the bound is that no variable
exceeds 100 in size.

## Measuring

[`measure`](@ref) pools runs into dwell and transition statistics.

```julia
m = measure(runs, nstates(spec))
m.dwell      # mean visit length per state
m.sem        # its standard error
m.counts     # counts[i, j]: transitions from i to j
m.samples    # every visit length, per state
```

Each run's first visit is left out because the run starts part-way through it, and its last because the run
ends before the visit does.

[`off_graph`](@ref) gives the share of transitions that aren't edges of the graph.

## Looking at the trajectory

`simulate` doesn't keep `u(t)`. To see the variables themselves, build the equation from the pieces every model
supplies and solve it with a solver from the SciML ecosystem; [A three-state ring](@ref) does this, and
[Internals and the backend interface](@ref) lists the pieces.
