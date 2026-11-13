# Closed-form dwell design from Horchler et al. (2015): the Stone–Holmes passage time past a saddle plus
# the deterministic transition, inverted for one shared rate α and a saddle value ν per state. The
# Horchler backend designs with it, and other backends seed from it.

"""
    stone_holmes_mean(δ, ε, λu, λs)

Leading-order mean passage time past a saddle with unstable and stable eigenvalues `λu` and `λs`,
neighbourhood size `δ` and noise `ε ≪ δ` (Stone & Holmes 1990):
`(2(log δ − log ε) + log 4λu − log1p(λu/λs) + γ) / 2λu`, with γ Euler's constant. This is
`passagetime` from SHCTools.jl.
"""
stone_holmes_mean(δ, ε, λu, λs) = (2(log(δ) - log(ε)) + log(4λu) - log1p(λu / λs) + Base.MathConstants.γ) / (2λu)

"""
Single-channel noise with the same mean escape time as racing channels with noise `εs` (half-normal
exit amplitudes; the SHCTools notebook's step 4).
"""
function effective_noise(εs)
    length(εs) == 1 && return float(only(εs))
    Z = [truncated(Normal(0, e), 0, Inf) for e in εs]
    top = log(maximum(εs)) + 8
    I, _ = quadgk(u -> prod(cdf(z, exp(u)) for z in Z), log(minimum(εs)) - 40, top; rtol=1e-8)
    return exp(top - I + (Base.MathConstants.γ + log(2)) / 2)
end

"""Neighbourhood edge in the form's own variable (`:x` is activity, `:u` its square root)."""
edge_size(form, κ, β) = form === :x ? κ * β : sqrt(κ * β)
"""Deterministic transition time between neighbourhoods at rate α."""
transition_time(form, κ, α) = (form === :x ? 2 : 1) * log(1 / κ - 1) / α

"""Designed mean dwell: the Stone–Holmes passage plus the transition."""
designed_dwell(form, α, ν, ε, β, κ) = stone_holmes_mean(edge_size(form, κ, β), ε, α / ν, α) + transition_time(form, κ, α)

"""
    dwell_rates(form, τ, ε_eff, β; κ=0.1, νmin=2.0) -> (α, ν)

One shared rate α and a saddle value ν per state, so each state's designed dwell is `τ[i]`. α is the
largest needed with ν at `νmin`; then each ν is solved by root finding. Throws an `ArgumentError` for
a dwell longer than the noise allows.
"""
function dwell_rates(form, τ, ε_eff, β; κ=0.1, νmin=2.0)
    # In α the designed dwell is ν/(2α)·(a + ln α), which peaks at α = e^(1−a) and falls below it, so the
    # search starts at the peak.
    function αᵢ(i, ν)
        f(logα) = designed_dwell(form, exp(logα), ν, ε_eff[i], β[i], κ) - τ[i]
        k = transition_time(form, κ, 1.0)
        a = 2log(edge_size(form, κ, β[i]) / ε_eff[i]) + Base.MathConstants.γ - log1p(1 / ν) + log(4 / ν) + 2k / ν
        lo = 1 - a
        f(lo) > 0 || throw(ArgumentError("state $i: dwell $(τ[i]) is longer than the $(round(f(lo) + τ[i]; sigdigits=3)) " *
            "that noise $(round(ε_eff[i]; sigdigits=2)) allows"))
        hi = lo + 1
        while f(hi) > 0
            hi += 2
        end
        return exp(find_zero(f, (lo, hi)))
    end
    α = maximum(αᵢ(i, νmin) for i in eachindex(τ))
    # The passage time ν/(2α)·(c − ln ν) peaks at ν = e^(c−1), so at a given noise and α there is a longest
    # dwell. The state that set α sits exactly at νmin, so the bracket starts below it.
    function νᵢ(i)
        f(v) = designed_dwell(form, α, v, ε_eff[i], β[i], κ) - τ[i]
        c = 2log(edge_size(form, κ, β[i]) / ε_eff[i]) + log(4α) + Base.MathConstants.γ
        top = min(1e6, exp(c - 1))
        f(top) > 0 || throw(ArgumentError("state $i: dwell $(τ[i]) is longer than the $(round(f(top) + τ[i]; sigdigits=3)) " *
            "that noise $(round(ε_eff[i]; sigdigits=2)) allows at the shared rate α = $(round(α; sigdigits=3))"))
        return find_zero(f, (νmin / 2, top))
    end
    return fill(α, length(τ)), νᵢ.(eachindex(τ))
end
