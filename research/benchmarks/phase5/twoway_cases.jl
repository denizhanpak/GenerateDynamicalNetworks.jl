# The two-way test graphs: four made up, and the worm graph measured from the tracks by worm_data.jl.
# Included by ctrnn_twoway.jl and by the Phase 5 notebook; needs backends.jl (for WORM_SPEC) loaded first.
const TWOWAY = [
    ("flip-flop", [0 1; 1 0.], [1.0, 0.3]),
    ("two-way and exit", [0 1 0; 0.4 0 0.6; 1 0 0], [2.0, 0.5, 1.2]),
    ("hub with two returns", [0 0.5 0.5; 1 0 0; 1 0 0], [1.0, 0.4, 0.8]),
    # the worm ring with Pause 1 → Forward added, at about the share in the raw tracks
    ("worm with return", [0 1 0 0; 0.47 0 0.53 0; 0 0 0 1; 1 0 0 0], [5.329, 0.441, 1.945, 0.208]),
]

"""The measured worm graph from `worm_data.jl`: every transition seen in the tracks, and the mean bouts."""
function worm_data_case()
    file = joinpath(@__DIR__, "results", "worm_data.csv")
    isfile(file) || return nothing
    P, τ = zeros(4, 4), zeros(4)
    for r in eachrow(readdlm(file, ','; header=true)[1])
        r[1] == "transition" && (P[Int(r[2]), Int(r[3])] = r[5])
        r[1] == "dwell" && (τ[Int(r[2])] = r[5])
    end
    return ("worm data", P, τ)
end
let w = worm_data_case()
    w === nothing || push!(TWOWAY, w)
end

twoway_spec(name) = let (_, P, τ) = only(c for c in TWOWAY if c[1] == name)
    (; P=float(P), τ, form=:u, ε₀=0.1, sample=startswith(name, "worm") ? WORM_SPEC.sample : minimum(τ) / 10)
end
