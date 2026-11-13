# A CTRNN fit as a `CTRNNs.jl` network, and a check that it behaves like the static-array version the
# search used: same weights, biases, time constants and gain, compared by dwell on independent runs.
# Run from notebooks/ with
#   julia --project=. -t 16 ../benchmarks/phase5/to_ctrnns.jl
include(joinpath(@__DIR__, "backends.jl"))
import CTRNNs

"""The fitted network as a `CTRNNs.CTRNN`; `CTRNNs` uses `W[i, j]` for j → i, as here."""
as_ctrnn(m::CTRNNModel) = CTRNNs.CTRNN(m.W; bias=m.b, tau=m.τ, gain=m.g)

function dwell_with_package(m, spec; T, seeds)
    net = as_ctrnn(m)
    every, dt = frame_steps(m, spec.sample, 0.01)
    runs = map(seeds) do seed
        # `CTRNNs.jl` writes τ dy = … + b dW, and the search adds ε dW to dy, so b = ε τ.
        prob = SDEProblem(net, Vector(ctrnn_start(m)), (0.0, T); noise=m.ε .* m.τ, seed=UInt64(seed))
        sol = solve(prob, EM(); dt, adaptive=false, saveat=spec.sample, maxiters=typemax(Int))
        state, t = Int[], Float64[]
        for (a, y) in zip(sol.t, sol.u)
            k = argmax(m.g .* (y .+ m.b))
            (isempty(state) || last(state) != k) && (push!(state, k); push!(t, a))
        end
        (; state, t)
    end
    return measure(runs, m.visible).dwell
end

_, P, τ = only(c for c in NAMED if c[1] == "worm ring")
spec = (; P, τ, form=:u, ε₀=0.1, sample=WORM_SPEC.sample)
m = ctrnn_model(spec, vec(readdlm(joinpath(@__DIR__, "results", "theta_ctrnn_oscillator_worm_ring_0.1.csv"), ',')))
ours = measure(simulate_frames(m; T=500.0, seeds=1:16, spec.sample, dtscale=0.01), 4).dwell
pkg = dwell_with_package(m, spec; T=500.0, seeds=101:116)
println("static version  dwell ", round.(ours; sigdigits=4))
println("CTRNNs.jl       dwell ", round.(pkg; sigdigits=4))
println("difference      ", round.(100 .* (pkg ./ ours .- 1); digits=1), " %")
