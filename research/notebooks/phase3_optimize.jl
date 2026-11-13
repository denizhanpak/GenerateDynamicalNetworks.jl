### A Pluto.jl notebook ###
# v1.1.0

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ c8a0b512-d45d-5bfd-8910-1baa9434aa89
begin
	import Pkg
	Pkg.activate(@__DIR__)   # notebooks/Project.toml
	Pkg.instantiate()
	using Plots, PlutoUI, DelimitedFiles, Printf
	# The packages the included files load. Pluto only re-imports what it sees in a cell, so they
	# are listed here too, or a cell rerun after a control moves can't find `solve` or `remake`.
	using SHCTools, StochasticDiffEq, StaticArrays, Graphs, Distributions, QuadGK, Roots
	using LinearAlgebra, Random, Statistics
	import CMAEvolutionStrategy as CES
	const phase3 = joinpath(@__DIR__, "..", "benchmarks", "phase3")
	include(joinpath(phase3, "worm.jl"))      # decoder, objective, search, validate, the worm problem
	include(joinpath(phase3, "..", "phase2", "cases.jl"))
end

# ╔═╡ fb9a729e-1eed-55d0-a41c-f6ffb8039158
md"""
# GenerateDynamicalNetworks: Phase 3, the optimisation layer

Phase 2's formula sets dwell times well at low noise. Phase 3 asks what to do when it doesn't, and it has one exit test: **starting from the formula, recover your worm fit (about 0.4% RMS at noise 0.1) and pass held-out validation.**

Choices you made for this phase:

| Choice | Setting |
|:--|:--|
| Decoder | sampled argmax |
| Search space | dwell requests through the formula, compared against raw ``\alpha`` and ``\nu`` |
| Off-graph transitions | penalised in the objective, 1% limit kept for acceptance |
| Branch term | ``\sum \log(f/p)^2`` |

The code is in `benchmarks/phase3/`. `optimize.jl` holds the decoder, objective, correction passes, CMA-ES, `validate`, `compare` and `fit`. `worm.jl` holds the worm problem. Searches are scripts whose results this notebook reads, and the controls below run small simulations live.
"""

# ╔═╡ 8e9523ec-5c70-5f88-a56b-f47e51a3500b
TableOfContents(title="Contents", depth=2)

# ╔═╡ 78080afa-a532-57f8-9a1e-52d5e2b6d244
begin
	const state_colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
	const muted = "#52514e"
	default(; fontfamily="sans-serif", linewidth=2, framestyle=:axes, grid=true, gridalpha=0.12,
		tickfontcolor=muted, guidefontcolor=muted, foreground_color_axis=muted,
		foreground_color_border=muted, legend_foreground_color=:transparent,
		legend_background_color=:transparent, size=(720, 320), left_margin=3Plots.mm,
		bottom_margin=3Plots.mm)
	sig(x, d=3) = x isa Number && isfinite(x) ? round(x; sigdigits=d) : "–"
	pct(x, d=1) = x isa Number && isfinite(x) ? "$(x > 0 ? "+" : "")$(round(100x; digits=d))%" : "–"
	mdtable(header, rows) = Markdown.parse(join(
		["| " * join(r, " | ") * " |" for r in [header, fill("---", length(header)), rows...]], "\n"))
	function table(name)
		data, header = readdlm(joinpath(phase3, "results", name), ','; header=true)
		keys = Tuple(Symbol.(vec(header)))
		cell(x) = x isa Number || x isa Bool ? x : something(tryparse(Float64, string(x)), string(x))
		[NamedTuple{keys}(Tuple(cell.(data[i, :]))) for i in axes(data, 1)]
	end
	theta(name) = vec(readdlm(joinpath(phase3, "results", "worm_$(name)_theta.csv"), ','))
	formula_seed = design(WORM_P, WORM_SPEC.τ; form=:u, ε₀=0.1)
	worm_models = ["your fit" => USER_FIT, "formula seed" => formula_seed,
		"CMA-ES on dwell requests" => request_model(WORM_SPEC, theta("request")),
		"CMA-ES on α and ν + polish" => raw_model(formula_seed, theta("raw_polished"))]
	istrue(x) = x === true || x == "true"
end

# ╔═╡ 89e692dc-7bab-50b0-8d0c-1d52a897eef3
md"""
## 1. The decoder

Sampled argmax reads the state on a frame clock: at each frame the current state is the one with the largest ``|u|``, and a bout lasts until another state wins a frame. The worm frame rate is 29.97 Hz. Each run's first and last bouts are dropped. A frame won by a hidden state keeps the last visible state.

Change the frame rate to see how much the dwell times depend on it. The last row of each model is the latch at 0.5, for comparison. This runs 16 × 500 s per row.
"""

# ╔═╡ 827a4161-ae9f-557e-9eae-c6ae398bc9f4
@bind fps_choice Select([29.97 => "29.97 Hz (the worm video)", 10.0 => "10 Hz", 120.0 => "120 Hz"]; default=29.97)

# ╔═╡ 550b3208-7576-54f7-b454-1735d44a1161
let
	seeds = 1:16
	rows = Any[]
	row(name, decoder, r) = vcat(name, decoder, [string(sig(d)) * " (" * pct(e) * ")" for (d, e) in zip(r.dwell, r.dwell ./ WORM_SPEC.τ .- 1)], pct(off_graph(r, WORM_P)))
	for (name, m) in worm_models[[1, 4]]
		spec = merge(WORM_SPEC, (; sample=1 / fps_choice))
		push!(rows, row(name, "argmax at $(fps_choice) Hz", measure(simulate_frames(m; T=500.0, seeds, spec.sample, dtscale=0.04), 4)))
		push!(rows, row(name, "latch at 0.5", measure(simulate(m; T=500.0, seeds, dt=0.04 / maximum(m.α)), 4)))
	end
	mdtable(["Model", "Decoder", WORM_STATES..., "Off-graph"], rows)
end

# ╔═╡ 6281d511-5d7a-5602-bd24-c1ae8f0a6b28
md"""
## 2. The objective

```math
f = \sum_i \log\!\left(\frac{m_i}{\tau_i}\right)^{2} + \sum_{\text{branches}} \log\!\left(\frac{f_{ij}}{p_{ij}}\right)^{2} + 5\,\max(\text{off} - 0.005,\ 0)
```

A state never visited costs 35, each dwell term is capped at 25, and a run that diverges costs 1000. The off-graph penalty has an allowance of 0.5%, half the acceptance limit, so a model already inside the limit isn't pushed away from its dwell targets to shave off the last few events.

Every evaluation uses seeds no earlier evaluation used, and the answer is CMA-ES's population mean. Phase 0 showed that fixed seeds overfit.

## 3. Correction and the formula's ceiling

Correction multiplies each requested dwell by target/measured and redesigns. At noise 0.1 this stops at once:
"""

# ╔═╡ 4312de27-bb92-5f9b-b173-b084a992f40b
let
	c = table("worm_correction.csv")
	mdtable(["Pass", "Request (F, P1, R, P2)", "Measured dwell", "Off-graph", "Note"],
		[[Int(r.pass), join(sig.([r.r1, r.r2, r.r3, r.r4]), ", "), join(sig.([r.d1, r.d2, r.d3, r.d4]), ", "), pct(r.off), r.note] for r in c])
end

# ╔═╡ 1827b99e-3560-5c15-9dda-1235638f1520
md"""
Forward comes out far short of its target, so the first correction asks for a dwell above what the formula can give at this noise. The leading-order passage time has a maximum in ``\nu``, and that maximum sets the ceiling.

## 4. Search

Both searches start from the formula seed, use 16 × 500 s per evaluation at ``dt = 0.04/\alpha``, and take the population mean. The dwell-request space has 4 dimensions and 208 evaluations. The raw space has one ``\alpha`` and one ``\nu`` per state, 8 dimensions and 804 evaluations. Each point is one evaluation on fresh noise, and the line is the running median of 25.
"""

# ╔═╡ ccd89b90-1b18-5b24-b4a9-d7291ecff605
let
	p = plot(; xlabel="evaluation", ylabel="objective", yscale=:log10, legend=:topright)
	for (name, file, col) in (("dwell requests", "worm_request_trace.csv", "#eb6834"), ("α and ν", "worm_raw_trace.csv", "#2a78d6"))
		d = readdlm(joinpath(phase3, "results", file), ',')
		scatter!(p, d[:, 1], d[:, 2]; ms=1.5, msw=0, alpha=0.35, color=col, label="")
		k = 25
		med = [median(d[i:i+k-1, 2]) for i in 1:size(d, 1)-k+1]
		plot!(p, d[k:end, 1], med; color=col, label=name)
	end
	p
end

# ╔═╡ 8553822b-c94c-5076-9463-6ccb4fc37e1e
md"""
## 5. The exit test

Every model below is scored on the same 64 held-out runs of 4000 s at ``dt = 0.01/\alpha``. The polish is three passes of ``\nu_i \leftarrow \nu_i\,\tau_i/m_i`` on long runs.
"""

# ╔═╡ 2d20fd7d-ccd4-591a-8d97-eb899758b1e8
let
	c = table("worm_compare.csv")
	rows = [[r.model; [string(sig(d, 4)) * " (" * pct(e) * ")" for (d, e) in zip((r.d1, r.d2, r.d3, r.d4), (r.e1, r.e2, r.e3, r.e4))]; pct(r.rms, 2); pct(r.off, 2); istrue(r.pass) ? "pass" : "fail"] for r in c]
	mdtable(["Model", WORM_STATES..., "RMS log error", "Off-graph", "Accepted"], rows)
end

# ╔═╡ cc1951cc-b1f6-56ce-83cb-10b4270510af
md"""
Standard errors for these runs are about 0.3% on Forward and 0.1–0.25% on the others, so differences of a few tenths of a percent are near the noise.

The fitted parameters, for reading the models side by side. In the u form, the escape rate from state ``k`` is ``\alpha_k/\nu_k``.
"""

# ╔═╡ 1e1dc536-a290-5248-abeb-ab301fcb7532
mdtable(["Model", "α", "ν", "Escape rate α/ν"],
	[[name, join(sig.(m.α), ", "), join(sig.(m.ν), ", "), join(sig.(m.α ./ m.ν), ", ")] for (name, m) in worm_models])

# ╔═╡ 3276bd20-7388-58c8-9d26-aa4de0ae388f
md"""
### Repeats

Each repeat starts from the same formula seed with a new CMA-ES seed and new noise, gets the same polish, and is scored on the same held-out runs as above. Repeat 3 was stopped at its 15-minute cap twice, the second time with ``\alpha`` bounded by 100, so it has no result. It wasn't stuck: a timed rerun was steadily completing about 1.5 evaluations per second with its largest ``\alpha`` near 86, and the step shrinks as ``\alpha`` grows.
"""

# ╔═╡ 2da581f9-ff48-54a6-b6e7-888a2388919d
let
	c = table("worm_compare.csv")
	find(name) = only(filter(r -> r.model == name, c))
	s, p, θ = find("CMA-ES on α and ν"), find("CMA-ES on α and ν + polish"), theta("raw_polished")
	rows = Any[Any[1, pct(s.rms, 2), pct(p.rms, 2), pct(maximum(abs, (p.e1, p.e2, p.e3, p.e4))), pct(p.off, 2), istrue(p.pass) ? "pass" : "fail", join(sig.(exp.(θ[1:4])), ", "), join(sig.(exp.(θ[5:8])), ", ")]]
	repeats = table("worm_repeats.csv")
	for rep in 2:4
		i = findfirst(x -> x.rep == rep, repeats)
		if i === nothing
			push!(rows, Any[rep, "–", "–", "–", "–", "stopped at the time cap", "–", "–"])
		else
			r = repeats[i]
			push!(rows, Any[rep, pct(r.search_rms, 2), pct(r.rms, 2), pct(maximum(abs, (r.e1, r.e2, r.e3, r.e4))), pct(r.off, 2), istrue(r.pass) ? "pass" : "fail", join(sig.([r.a1, r.a2, r.a3, r.a4]), ", "), join(sig.([r.n1, r.n2, r.n3, r.n4]), ", ")])
		end
	end
	mdtable(["Run", "RMS after search", "RMS after polish", "Worst dwell error", "Off-graph", "Accepted", "α", "ν"], rows)
end

# ╔═╡ 11d3ee77-ca15-536d-8328-0a1bbe8a10af
md"""
### Watch one

Activity ``x = u^2`` for one run, with the decoded state underneath. The noise is 0.1 for all of them.
"""

# ╔═╡ 7a189203-0b8e-508c-8008-fccbb8ef13e2
@bind model_choice Select([i => first(worm_models[i]) for i in eachindex(worm_models)]; default=1)

# ╔═╡ 72ffe8b4-0289-5763-ac74-66d415657c32
@bind watch_seed Select([1 => "seed 1", 2 => "seed 2", 3 => "seed 3"]; default=1)

# ╔═╡ 8e5c7cc5-aa8f-5367-9416-61adde04b49a
let
	name, m = worm_models[model_choice]
	every, dt = frame_steps(m, WORM_SPEC.sample, 0.02)
	prob = remake(sde_problem(m; T=60.0); seed=UInt64(watch_seed))
	sol = solve(prob, EM(); dt, adaptive=false, saveat=WORM_SPEC.sample, maxiters=typemax(Int))
	x = reduce(hcat, [Vector(u .^ 2) for u in sol.u])'
	labels = [argmax(abs.(u)) for u in sol.u]
	top = plot(; ylabel="activity x", title=name, titlefontsize=11, legend=:outerright, xticks=false)
	for i in 1:4
		plot!(top, sol.t, x[:, i]; color=state_colors[i], lw=1.2, label=WORM_STATES[i])
	end
	bottom = plot(sol.t, labels; seriestype=:steppost, color=muted, lw=1.2, label="", xlabel="time (s)",
		yticks=(1:4, WORM_STATES), ylims=(0.5, 4.5))
	plot(top, bottom; layout=Plots.grid(2, 1; heights=[0.65, 0.35]), size=(760, 430))
end

# ╔═╡ 0d07d83d-06fe-5ebb-9196-40e23aec47a9
md"""
## 6. Branches and hidden states

The worm has no branches, so the branch term and hidden states are tested on two other graphs through the generic pipeline `fit`: formula seed, up to 4 correction passes, and CMA-ES on raw rates only if those don't pass. The frame interval here is one tenth of the shortest target dwell. Noise per state keeps the formula's ratios, and the search doesn't change it.
"""

# ╔═╡ 83c180a2-a3a3-529e-9ce7-5cd153b4e7ad
let
	c = table("branching.csv")
	mdtable(["Graph", "Noise", "Solved by", "Hidden", "Worst dwell error", "Off-graph", "Branches", "Accepted", "Evaluations", "Seconds"],
		[[r.case, r.noise, r.stage, Int(r.hidden), pct(r.worst_err), pct(r.off, 2), istrue(r.branches_ok) ? "ok" : "miss", istrue(r.pass) ? "pass" : "fail", Int(r.evals), round(Int, r.seconds)] for r in c])
end

# ╔═╡ 72b0cba9-f6e9-57d3-90c0-68526ee868e4
let
	c = table("branching_edges.csv")
	mdtable(["Graph", "Noise", "Branch", "Designed", "Measured", "Inside 95% CI", "Second held-out set"],
		[[r.case, r.noise, "$(Int(r.from)) → $(Int(r.to))", r.designed, sig(r.measured), istrue(r.within_ci) ? "yes" : "no", sig(r.repeat_measured)] for r in c])
end

# ╔═╡ Cell order:
# ╠═c8a0b512-d45d-5bfd-8910-1baa9434aa89
# ╠═fb9a729e-1eed-55d0-a41c-f6ffb8039158
# ╠═8e9523ec-5c70-5f88-a56b-f47e51a3500b
# ╠═78080afa-a532-57f8-9a1e-52d5e2b6d244
# ╠═89e692dc-7bab-50b0-8d0c-1d52a897eef3
# ╠═827a4161-ae9f-557e-9eae-c6ae398bc9f4
# ╠═550b3208-7576-54f7-b454-1735d44a1161
# ╠═6281d511-5d7a-5602-bd24-c1ae8f0a6b28
# ╠═4312de27-bb92-5f9b-b173-b084a992f40b
# ╠═1827b99e-3560-5c15-9dda-1235638f1520
# ╠═ccd89b90-1b18-5b24-b4a9-d7291ecff605
# ╠═8553822b-c94c-5076-9463-6ccb4fc37e1e
# ╠═2d20fd7d-ccd4-591a-8d97-eb899758b1e8
# ╠═cc1951cc-b1f6-56ce-83cb-10b4270510af
# ╠═1e1dc536-a290-5248-abeb-ab301fcb7532
# ╠═3276bd20-7388-58c8-9d26-aa4de0ae388f
# ╠═2da581f9-ff48-54a6-b6e7-888a2388919d
# ╠═11d3ee77-ca15-536d-8328-0a1bbe8a10af
# ╠═7a189203-0b8e-508c-8008-fccbb8ef13e2
# ╠═72ffe8b4-0289-5763-ac74-66d415657c32
# ╠═8e5c7cc5-aa8f-5367-9416-61adde04b49a
# ╠═0d07d83d-06fe-5ebb-9196-40e23aec47a9
# ╠═83c180a2-a3a3-529e-9ce7-5cd153b4e7ad
# ╠═72b0cba9-f6e9-57d3-90c0-68526ee868e4
