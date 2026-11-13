# Phase 7 static checks, no simulation: (1) each vertex's Jacobian matches the designed rates; (2) which
# graphs the pairwise design can lay out, with how many neurons and transits, and the race seed.
# Run from notebooks/ with  julia --project=. -t 1 ../benchmarks/phase7/precheck.jl
include(joinpath(@__DIR__, "hopfield.jl"))
include(joinpath(@__DIR__, "..", "phase5", "twoway_cases.jl"))

spec = (; P=float(ring(3)), τ=[1.0, 3.0, 10.0], form=:u, ε₀=0.1, sample=0.1)
L = hop_layout(spec)
rough, c = rough_rates(spec)
m, worst = hop_model(spec, L, rough; fast=c, contract=c)
q = hop_params(m)
gap = 0.0
for v in 1:2^m.n
    ψ = SVector{m.n}(ntuple(j -> corner_bit(v, j) == 1 ? Float64(π) : 0.0, m.n))
    ev = sort(real.(eigvals(jac(drift_hop, q, ψ))))
    global gap = max(gap, maximum(abs, ev .- sort([vertex_rate(m.J, m.b, m.k, v, j) for j in 1:m.n])), maximum(abs, drift_hop(ψ, q, 0.0)))
end
@printf("3-ring: %d neurons, design violation %.1e, largest gap Jacobian vs design rates %.1e\n", m.n, worst, gap)
println("  rates at used vertices along their path step: ", [(corner_name(v, m.n), j, round(vertex_rate(m.J, m.b, m.k, v, j); sigdigits=3)) for (v, j, e, s) in path_steps(L)])

worm = worm_data_case()
cases = [[(name, P, τ, ε) for ε in (1e-3, 0.1) for (name, P, τ) in NAMED]; [(worm[1], worm[2], worm[3], 0.1)]]
for (name, P, τ, ε) in cases
    sp = (; P=float(P), τ, form=:u, ε₀=ε, sample=minimum(τ) / 10)
    t = @elapsed L = try
        hop_layout(sp)
    catch err
        err isa ArgumentError || rethrow(); nothing
    end
    if L === nothing
        @printf("%-22s ε=%-6g no layout up to 9 neurons (%.0f s)\n", name, ε, t)
        continue
    end
    seed, c = hop_seed(sp, L)
    _, w = hop_model(sp, L, seed; fast=c, contract=c)
    @printf("%-22s ε=%-6g %d neurons, %d transits, violation at rule-of-thumb ok, at race seed %.1e, seed %s (%.0f s)\n",
            name, ε, L.M, transits(L), w, round.(seed; sigdigits=3), t)
end
