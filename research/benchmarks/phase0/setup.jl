import Pkg
Pkg.activate(@__DIR__)
want = ["CMAEvolutionStrategy","Evolutionary","Metaheuristics","BlackBoxOptim","SimpleDiffEq","StochasticDiffEq","DiffEqGPU","KernelAbstractions","StaticArrays","BenchmarkTools","Statistics","Random","LinearAlgebra","Printf","Roots","DelimitedFiles"]
for w in want
    try
        Pkg.add(w; io=devnull); println("ok   ", w)
    catch e
        println("FAIL ", w, " :: ", first(replace(sprint(showerror, e), "\n"=>" "), 260))
    end
end
Pkg.status()
