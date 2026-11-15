```@meta
CurrentModule = GenerateDynamicalNetworks
```

# Dynamical classes

Two models can make the same sequence of visits by different means. [`classify`](@ref) reports which, by
looking at what the system does when the noise is switched off.

| Class | Without noise the system… | Noise is needed to… |
|:--|:--|:--|
| `:heteroclinic` | approaches a saddle and stalls there, or cycles ever more slowly | leave each state |
| `:excitable` | settles at a stable equilibrium | cross a threshold to the next state |
| `:ghost` | cycles on its own, with long slow passages | nothing; it sets the variability only |
| `:limit_cycle` | cycles on its own at an even pace | nothing; it sets the variability only |
| `:unknown` | does none of these consistently | |

A ghost and a limit cycle are both periodic orbits. The difference is how uneven the speed is along the orbit:
a ghost passes close to where an equilibrium has just disappeared and nearly stops there.

## How the class is found

1. One noisy run is made at the specification's noise, of length 40 times the sum of the dwells.
2. Five starting points are taken: the model's own start and four points from the second half of the noisy
   run.
3. From each, the system is run without noise for 150 times the sum of the dwells, and the end of the run is
   examined.

   | The noise-free run ends… | Outcome | Class |
   |:--|:--|:--|
   | at rest, at a point with no unstable direction | `stable` | excitable |
   | at rest, at a point with an unstable direction | `saddle` | heteroclinic |
   | in a repeating cycle of states, each cycle over 1.2 times as long as the one before | `growing` | heteroclinic |
   | in a repeating cycle whose peak speed is at least 100 times its slowest | `periodic` | ghost |
   | in a repeating cycle with a smaller ratio of speeds | `periodic` | limit cycle |
   | otherwise | `irregular` | unknown |

4. If all five starting points give the same class, that is the model's class. If they disagree it is
   `:unknown`.

```julia
c = classify(model, spec)
c.class        # :heteroclinic
c.outcomes     # "saddle saddle saddle saddle saddle"
```

`outcomes` lists what each starting point's run did, with the speed ratio of periodic runs in brackets. It is
the first thing to read when the class is `:unknown`.

The run lengths, the number of starting points and the ghost ratio are keyword arguments. A second method
classifies any vector field with a decoder, without a model.

## What the class depends on

The class belongs to the fitted model, not to the backend alone. The CTRNN backend, for example, gives
excitable models on some of the benchmark networks and ghosts on others ([Results](@ref)).

`:unknown` doesn't mean the model is wrong. It appears, for instance, when a noise-free run is too slow to
complete enough cycles in the time allowed, or when starting points on different parts of the orbit end
differently.
