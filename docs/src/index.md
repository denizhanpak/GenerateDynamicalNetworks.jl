```@meta
CurrentModule = GenerateDynamicalNetworks
```

# GenerateDynamicalNetworks

Given a transition graph and the mean time spent in each state, this package builds a noisy dynamical system
whose switching between states reproduces both. You say which states follow which, with what probability, and
how long each lasts on average; a *backend* builds the system, and the package then measures it against what
you asked for.

## Installation

The package isn't registered yet. Install it from its repository:

```julia
using Pkg
Pkg.add(url="https://github.com/denizhanpak/GenerateDynamicalNetworks.jl")
```

## A short example

```julia
using GenerateDynamicalNetworks

P = [0 1 0; 0 0 1; 1 0 0]                    # state 1 → 2 → 3 → 1
spec = TransitionSpec(P, [1.0, 3.0, 10.0]; noise=1e-3)

model = fit(spec, Horchler())                # a Lotka–Volterra network
accept(model, spec)                          # dwells and branches on held-out runs
classify(model, spec)                        # (class = :heteroclinic, …)
runs = simulate(model, spec; T=500)          # the states visited and when
```

[A three-state ring](@ref) goes through this example step by step.

## Backends

Seven constructions are included. Each is a type passed to [`fit`](@ref).

| Backend | Type | The system it builds |
|:--|:--|:--|
| [Horchler](@ref horchler-backend) | [`Horchler`](@ref) | Lotka–Volterra network, one node per state, designed in closed form |
| [Simplex](@ref simplex-backend) | [`Simplex`](@ref) | one cell per state with cubic coupling |
| [Cylinder](@ref cylinder-backend) | [`Cylinder`](@ref) | one position variable for the state and one cell per edge |
| [Two-population](@ref twopop-backend) | [`TwoPopulation`](@ref) | one cell per state and one per edge, in two coupled populations |
| [CTRNN design](@ref ctrnn-backend) | [`CTRNNDesign`](@ref) | continuous-time recurrent neural network with relay cells |
| [Synchrony](@ref synchrony-backend) | [`Synchrony`](@ref) | pairs of phase oscillators, with states at patterns of in-phase and anti-phase pairs |
| [Hopfield](@ref hopfield-backend) | [`Hopfield`](@ref) | Hopfield network with states at the vertices of a cube |

Each backend is labelled *stable* or *experimental* by how it does on the benchmark suite: every backend is
run on the example networks at two noise levels and put through the acceptance protocol
([The acceptance rule](@ref)). The table is built from the stored runs; [Results](@ref) has every case.

```@eval
Main.ResultsTables.status()
```

## Where to go next

- **Tutorials** work through three networks from specification to accepted model.
- **Concepts** explains the specification, the simulator, the acceptance rule and the dynamical classes.
- **Backends** describes each construction: its equations, how it is designed and where it fails.
- **Reference** lists every exported function, and the interface a new backend implements.
