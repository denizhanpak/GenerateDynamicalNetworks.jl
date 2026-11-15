```@meta
CurrentModule = GenerateDynamicalNetworks
```

# Results

Every backend was run on the example networks and put through the acceptance protocol
([The acceptance rule](@ref)). The tables on this page are built from the stored runs in `benchmarks/results/`
each time the documentation is built.

```@eval
Main.ResultsTables.provenance()
```

## The suite

Each case is one backend on one network at one noise level.

- Most backends run the six networks with one-way edges ([Example networks](@ref)) at noise 0.001 and 0.1.
- Synchrony and Hopfield also run a three-state ring with dwells 1, 3 and 10.
- The CTRNN backend runs all ten networks, including the four with two-way edges, at noise 0.1 only.

A case passes when the fitted model passes all three checks. Times are for the whole suite on that backend:
all its fits, and all its checks.

```@eval
Main.ResultsTables.summary()
```

To run the suite yourself:

```
julia --project=benchmarks -t 14 benchmarks/suite.jl
```

## How the cases fail

A case can fail one, two or three of its checks. Failing one check by a small margin is what the branch rule
does to a correct model from time to time ([How strict the branch rule is](@ref)); failing all three points to
a consistent error.

```@eval
Main.ResultsTables.failures()
```

## Reproduction of the prototype

The package was ported from a research prototype. To check the port, each backend was refitted with the
prototype's settings on one network and compared with the prototype's stored result.

- **Equal numbers:** the refit follows the same random-number path, so every stored number must be equal.
- **Same band verdicts:** the refit is a different model (one solver setting was fixed in the Hopfield backend,
  and the prototype's special layout for the three-state ring wasn't ported to the synchrony backend). Every
  state's mean dwell must then be on the same side of the 5% band as in the stored result, on each check.

```@eval
Main.ResultsTables.reproduction()
```

The Hopfield refit on the Δ-clique agrees on the dwells but fails branch checks that the prototype's fit
passed.

## Every case

"Worst dwell error" and "Off graph" are from set 1. "Class" is the result of [`classify`](@ref).

### Synchrony

```@eval
Main.ResultsTables.cases("synchrony")
```

### Hopfield

```@eval
Main.ResultsTables.cases("hopfield")
```

### CTRNN design

```@eval
Main.ResultsTables.cases("ctrnn")
```

### Horchler

```@eval
Main.ResultsTables.cases("horchler")
```

### Synchrony, race seed

```@eval
Main.ResultsTables.cases("synchrony_race")
```

### Cylinder

```@eval
Main.ResultsTables.cases("cylinder")
```

### Simplex

```@eval
Main.ResultsTables.cases("simplex")
```

### Two-population

```@eval
Main.ResultsTables.cases("twopop")
```
