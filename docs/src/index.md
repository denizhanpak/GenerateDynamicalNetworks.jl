```@meta
CurrentModule = GenerateDynamicalNetworks
```

# GenerateDynamicalNetworks

Given a transition graph and the mean time spent in each state, build a noisy dynamical system whose
switching between states reproduces both. The package is being ported from a research prototype; this
version has the core layer that every construction ("backend") shares:

- [`TransitionSpec`](@ref): the transition matrix, mean dwells, noise and the decoder's frame interval,
  checked by [`spec_problems`](@ref); [`example`](@ref) gives ten example networks.
- [`simulate`](@ref): one Euler–Maruyama simulator for every backend, reading the state once per frame.
  A run depends only on its seed.
- [`measure`](@ref), [`verdict`](@ref), [`validate`](@ref) and [`accept`](@ref): dwell and branch
  statistics and the acceptance rule. Every mean dwell must be within 5%, every branch inside its 95%
  interval, and under 1% of transitions off the graph. This must hold on two sets of held-out seeds
  and at half the step.
- [`classify`](@ref): the dynamical class of a model (heteroclinic, excitable, ghost, limit cycle or
  unknown).

A backend's model subtypes [`AbstractNetworkModel`](@ref) and implements a handful of methods (its
drift, parameters, starting state, fastest rate and frame decoder); all of the above then works on it.

```julia
using GenerateDynamicalNetworks
spec = example("foraging"; noise=0.1)
spec.P, spec.τ
```

The constructions themselves (Horchler Lotka–Volterra, simplex, cylinder, two-population, CTRNN,
phase oscillators, Hopfield) are added backend by backend.
