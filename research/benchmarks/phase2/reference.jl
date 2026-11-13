# Runs the SHCTools notebook as a plain script (default control values) and saves the
# numbers Phase 2 must reproduce. Run with --project=../../../SHCTools.jl/notebooks
module Ref
# SHCTools.jl checked out beside this repository (see research/README.md)
include(joinpath(@__DIR__, "..", "..", "..", "..", "SHCTools.jl", "notebooks", "dwell_times_to_simulation.jl"))
end
using DelimitedFiles
R = Ref
out = joinpath(@__DIR__, "reference")
mkpath(out)
s = R.system
writedlm(joinpath(out, "design.csv"), [["state" "tau" "eps" "eps_eff" "alpha" "nu"]; hcat(R.states, s.τ[1:5], s.ε[1:5], s.ε_eff[1:5], s.α[1:5], s.ν[1:5])], ',')
writedlm(joinpath(out, "rho.csv"), s.ρ, ',')
m, r = R.measured, R.remeasured
writedlm(joinpath(out, "table.csv"), [["state" "target" "first" "first_sem" "first_n" "corrected" "corrected_sem"];
    hcat(R.states, R.τ, m.dwell, m.sem, length.(m.samples), r.dwell, r.sem)], ',')
writedlm(joinpath(out, "counts_first.csv"), m.counts, ',')
writedlm(joinpath(out, "counts_corrected.csv"), r.counts, ',')
println("first: ", round.(m.dwell; digits=4), "\ncorrected: ", round.(r.dwell; digits=4))
println("left fraction: ", m.counts[2, 3] / sum(m.counts[2, :]), "  corrected: ", r.counts[2, 3] / sum(r.counts[2, :]))
