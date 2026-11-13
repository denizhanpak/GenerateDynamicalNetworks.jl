# The race seed for the synchrony and Hopfield backends: each exit of a state is a 1-D diffusion of a
# phase difference ψ, and the exits race independently.
#
# Leaving a corner along one direction with departure rate λ and arrival contraction `contract`,
#     dψ = sin ψ (g + k cos ψ) dt + √2 η dW,     g + k = λ,  k − g = −contract,
# from ψ = 0, absorbed at |ψ| = π/2 (the decoder's boundary). Each exit's survival S_e(t) comes from a
# continuous-time chain on a grid with Scharfetter–Gummel rates (stable for strong drift); then
# dwell = ∫ Π S_j dt and share_e = ∫ f_e Π_{j≠e} S_j dt with f_e = −S_e'. Full distributions matter:
# a fast state's main exit is close to a fixed delay, so a slower exit almost never wins it.

"""The grid generator (tridiagonal) of one exit's ψ on (−π/2, π/2): Scharfetter–Gummel rates, both ends absorbing."""
function exit_generator(λ, contract, η; n=241)
    g, k = (λ + contract) / 2, (λ - contract) / 2
    D = η^2
    ψ = range(-π / 2, π / 2; length=n + 2)[2:end-1]          # interior points
    dx = step(ψ)
    f(x) = sin(x) * (g + k * cos(x))
    B(z) = abs(z) < 1e-8 ? 1.0 - z / 2 : z / expm1(z)
    up = [D / dx^2 * B(-f(ψ[i] + dx / 2) * dx / D) for i in 1:n]
    down = [D / dx^2 * B(f(ψ[i] - dx / 2) * dx / D) for i in 1:n]
    return Tridiagonal(down[2:end], -(up .+ down), up[1:end-1])
end

"""
    race(λs, contract, η; n=241, dt0=1e-4, grow=1.002, tail=1e-9) -> (dwell, shares)

Mean dwell and branch shares of independent racing exits with departure rates `λs` and contraction
`contract` (one value, or one per exit), at noise `η`. Each exit's survival from ψ = 0 solves the
backward equation by implicit Euler on a time grid that starts at `dt0` and grows by `grow` per step,
until the joint survival falls below `tail`; the integrals are trapezoid sums on the same grid.
"""
function race(λs, contract, η; n=241, dt0=1e-4, grow=1.002, tail=1e-9)
    Qs = [exit_generator(λ, contract isa Number ? contract : contract[e], η; n) for (e, λ) in enumerate(λs)]
    k, mid = length(λs), (n + 1) ÷ 2
    us = [ones(n) for _ in 1:k]
    S = ones(k)
    dwell, share = 0.0, zeros(k)
    dt = dt0
    while prod(S) > tail
        S_old = copy(S)
        for e in 1:k
            us[e] = (I - dt * Qs[e]) \ us[e]
            S[e] = us[e][mid]
        end
        all_old, all_new = prod(S_old), prod(S)
        dwell += dt * (all_old + all_new) / 2
        for e in 1:k
            others = prod((S_old[j] + S[j]) / 2 for j in 1:k if j != e; init=1.0)
            share[e] += (S_old[e] - S[e]) * others
        end
        dt *= grow
    end
    return dwell, share ./ sum(share)
end

"""`[−log dwell; log share ratios]` of the race, the outputs the out-rate correction works on."""
race_outputs(λs, contract, η) = (r = race(λs, contract, η); k = length(λs);
    [-log(r[1]); log.(max.(r[2][1:k-1], 1e-300)) .- log(max(r[2][k], 1e-300))])

"""
    race_seed_rates(spec, start, contract, η; iters=40) -> out

Out-rates (one per edge, in `edge_list` order) that make every state's race give its dwell and branch
shares, by damped Newton in λ from `start` (at most 2 per step, within ±`contract`).
"""
function race_seed_rates(spec, start, contract, η; iters::Integer=40)
    n, edges = nstates(spec), edge_list(spec.P)
    out = copy(start)
    for i in 1:n
        o = [e for (e, (a, _)) in enumerate(edges) if a == i]
        k = length(o)
        want = [spec.P[i, edges[e][2]] for e in o]
        goal = [-log(spec.τ[i]); log.(want[1:k-1]) .- log(want[k])]
        λ = out[o]
        for _ in 1:iters
            y = race_outputs(λ, contract, η)
            maximum(abs, y .- goal) < 1e-4 && break
            J = zeros(k, k)
            for j in 1:k
                d = zeros(k); d[j] = 1e-4 * max(1.0, abs(λ[j]))
                J[:, j] = (race_outputs(λ .+ d, contract, η) .- race_outputs(λ .- d, contract, η)) ./ (2d[j])
            end
            step = J \ (goal .- y)
            scale = min(1.0, 2.0 / maximum(abs, step))           # at most 2 per step in λ
            λ = clamp.(λ .+ scale .* step, -contract, contract)
        end
        out[o] = λ
    end
    return out
end

"""
    race_response(contract, η)

A starting Broyden matrix for [`correct_rates`](@ref) from the race, by central differences in the
correction's `asinh(λ/c)` coordinates: a function `(o, out, c) -> J`.
"""
function race_response(contract, η)
    return function (o, out, c)
        k = length(o)
        x = asinh.(out[o] ./ c)
        J = zeros(k, k)
        for j in 1:k
            d = zeros(k); d[j] = 1e-3
            J[:, j] = (race_outputs(c .* sinh.(x .+ d), contract, η) .- race_outputs(c .* sinh.(x .- d), contract, η)) ./ 2e-3
        end
        return J
    end
end
