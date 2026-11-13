# Phase 6, option B (time-boxed): seed each exit's departure rate from a 1-D first-passage time, as the
# CTRNN edge design does, instead of the rule of thumb law/τ·√(p·k).
#
# Leaving a state corner along pair σ, the phase difference obeys
#     dψ = sin ψ (g + k cos ψ + h sin² ψ) dt + √2 η dW,
# with g + k = λ (departure rate), k − g = −contract (arrival), and h an optional third-harmonic term
# that sets the cubic at the departure end without moving either rate. ψ ↦ −ψ is the same corner, so
# the exit is |ψ| reaching π/2 (the decoder's boundary): reflecting at 0, absorbing at π/2. If the
# exits of a state race independently and each is near-exponential, exit e's rate is 1/MFPT_e, so
# MFPT_e = τ_i / p_e gives both the dwell and the branch shares.
#
# Not included by suite.jl; nothing here changes the running suite.
include(joinpath(@__DIR__, "synchrony.jl"))
using Roots

"""∫₀^y sin ψ (g + k cos ψ + h sin² ψ) dψ."""
drift_integral(y, g, k, h) = g * (1 - cos(y)) + k * sin(y)^2 / 2 + h * ((1 - cos(y)) - (1 - cos(y)^3) / 3)

"""Log of the mean time for |ψ| to go from 0 to π/2, with noise √2 η on ψ (D = η²), in log space so
deep barriers don't overflow."""
function log_exit_time(λ, contract, η; h=0.0, n=4000)
    g, k = (λ + contract) / 2, (λ - contract) / 2
    D = η^2
    ys = range(0.0, π / 2; length=n + 1)
    Φ = [-drift_integral(y, g, k, h) / D for y in ys]
    w = step(ys)
    lae(a, b) = max(a, b) + log1p(exp(-abs(a - b)))
    inner = accumulate(lae, -Φ)                     # log ∫₀^y e^{−Φ}, up to the factor w
    # T = (1/D) ∫₀^{π/2} e^{Φ(y)} ∫₀^y e^{−Φ(z)} dz dy
    return reduce(lae, Φ .+ inner) + 2log(w) - log(D)
end
exit_time(λ, contract, η; kw...) = exp(log_exit_time(λ, contract, η; kw...))

"""The departure rate λ whose 1-D exit time is `T`, within [lo, hi]; clamped to the ends."""
function rate_for(T, contract, η; h=0.0, lo=-contract, hi=contract)
    f(λ) = log_exit_time(λ, contract, η; h) - log(T)
    f(lo) <= 0 && return lo
    f(hi) >= 0 && return hi
    return find_zero(f, (lo, hi), Bisection(); xatol=1e-8)
end

"""Seeded out-rates for every edge (in `edge_list` order) and the contraction they were solved at."""
function seed_out_1d(spec; contract=nothing)
    edges = edge_list(spec.P)
    _, c4 = rough_rates(spec)
    c = contract === nothing ? c4 : contract
    out = [rate_for(spec.τ[i] / spec.P[i, j], c, spec.ε₀) for (i, j) in edges]
    return out, c
end

"""
A state's starting Broyden matrix from the 1-D race: inputs asinh(λ_e/c) over its out-edges, outputs
log total exit rate and log rate ratios, with rate_e = 1/exit time_e. Central differences.
"""
function race_response(contract, η)
    return function (o, out, c)
        k = length(o)
        x = asinh.(out[o] ./ c)
        y(x) = (lr = [-log_exit_time(c * sinh(xi), contract, η) for xi in x];
                top = maximum(lr); [top + log(sum(exp.(lr .- top))); lr[1:k-1] .- lr[k]])
        J = zeros(k, k)
        for j in 1:k
            d = zeros(k); d[j] = 1e-4
            J[:, j] = (y(x .+ d) .- y(x .- d)) ./ 2e-4
        end
        return J
    end
end

# ── The race with full exit-time distributions ───────────────────────────────────────────────
# The mean-only race above assumes exponential exits. A fast state's main exit is close to a fixed
# delay, so a slower exit almost never wins it, and the shares come out wrong. Here each exit's
# survival S_e(t) comes from the 1-D diffusion on (−π/2, π/2), absorbed at both ends (|ψ| = π/2), as a
# continuous-time chain on a grid with Scharfetter–Gummel rates (stable for strong drift). Exits race
# independently from ψ = 0: dwell = ∫ Π S_j dt, share_e = ∫ f_e Π_{j≠e} S_j dt with f_e = −S_e'.

"""The grid generator (tridiagonal) of one exit's ψ: Scharfetter–Gummel rates, both ends absorbing."""
function exit_generator(λ, contract, η; h3=0.0, n=241)
    g, k = (λ + contract) / 2, (λ - contract) / 2
    D = η^2
    ψ = range(-π / 2, π / 2; length=n + 2)[2:end-1]          # interior points
    dx = step(ψ)
    f(x) = sin(x) * (g + k * cos(x) + h3 * sin(x)^2)
    B(z) = abs(z) < 1e-8 ? 1.0 - z / 2 : z / expm1(z)
    up = [D / dx^2 * B(-f(ψ[i] + dx / 2) * dx / D) for i in 1:n]
    down = [D / dx^2 * B(f(ψ[i] - dx / 2) * dx / D) for i in 1:n]
    return Tridiagonal(down[2:end], -(up .+ down), up[1:end-1])
end

"""
Dwell and shares of independent racing exits with these departure rates. Each exit's survival from
ψ = 0 solves u' = Q u, u(0) = 1 (backward equation) by implicit Euler on a time grid that starts
fine and grows; the integrals are trapezoid sums on the same grid.
"""
function race(λs, contract, η; n=241, dt0=1e-4, grow=1.002, tail=1e-9)
    # `contract` is one value for every exit or one per exit
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

race_outputs(λs, contract, η) = (r = race(λs, contract, η); k = length(λs);
    [-log(r[1]); log.(max.(r[2][1:k-1], 1e-300)) .- log(max(r[2][k], 1e-300))])

"""Seed every state's out-rates by solving the race for its dwell and shares (damped Newton in λ)."""
function seed_out_race(spec; contract=nothing, iters=40, start=nothing)
    n, edges = size(spec.P, 1), edge_list(spec.P)
    # `start` replaces the mean-only start, whose uniform grid can't resolve the boundary layer at low
    # noise (at 1e-3 it gives about the same exit time for every rate)
    out, c = seed_out_1d(spec; contract)
    start === nothing || (out = copy(start))
    for i in 1:n
        o = [e for (e, (a, _)) in enumerate(edges) if a == i]
        k = length(o)
        want = [spec.P[i, edges[e][2]] for e in o]
        goal = [-log(spec.τ[i]); log.(want[1:k-1]) .- log(want[k])]
        λ = out[o]
        for _ in 1:iters
            y = race_outputs(λ, c, spec.ε₀)
            maximum(abs, y .- goal) < 1e-4 && break
            J = zeros(k, k)
            for j in 1:k
                d = zeros(k); d[j] = 1e-4 * max(1.0, abs(λ[j]))
                J[:, j] = (race_outputs(λ .+ d, c, spec.ε₀) .- race_outputs(λ .- d, c, spec.ε₀)) ./ (2d[j])
            end
            step = J \ (goal .- y)
            scale = min(1.0, 2.0 / maximum(abs, step))           # at most 2 per step in λ
            λ = clamp.(λ .+ scale .* step, -c, c)
        end
        out[o] = λ
    end
    return out, c
end

"""Starting Broyden matrix from the full race, in the fit's asinh(λ/c) coordinates."""
function race_response_full(contract, η)
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
