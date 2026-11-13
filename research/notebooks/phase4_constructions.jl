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

# ╔═╡ 14f808ff-1f8b-571a-9d67-f80a4deb04e2
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
	const phase4 = joinpath(@__DIR__, "..", "benchmarks", "phase4")
	const phase3results = joinpath(phase4, "..", "phase3", "results")
	include(joinpath(phase4, "backends.jl"))   # both backends, plus Phase 2 and 3 through it
end

# ╔═╡ 0b196ab5-3d19-53cc-aae0-1b31452f8e38
md"""
# GenerateDynamicalNetworks: Phase 4, simplex and cylinder

Two more constructions, both from Ashwin & Postlethwaite (2013), plugged into the Phase 3 pipeline. The exit test from the plan: **both pass the test graphs, or report which graphs they can't support.**

Choices you made for this phase:

| Choice | Setting |
|:--|:--|
| Simplex time scale | one global speed, as well as a rate per edge and a contraction per state |
| Simplex branches | one shared noise value, branch odds set by the edge rates, no hidden states |
| Cylinder | one global speed, a rate ``L_\alpha`` per edge, ``k_\alpha = k_\omega = 2``, ``L_\omega = 1.5625`` |
| Cylinder decoder | a frame is vertex ``k`` only if ``|p-k| < 0.25`` and every ``|y| < 0.5``, otherwise the last vertex holds |
| Test suite | the six Phase 2 graphs at noise ``10^{-3}`` and ``0.1``, with the Phase 3 acceptance rules |

The code is in `benchmarks/phase4/`. `backends.jl` holds both models and the pipeline, and `suite.jl` runs the fits that this notebook reads. The control at the end simulates one fitted model live.
"""

# ╔═╡ 6070479e-b106-539d-98b3-d9370310ded0
TableOfContents(title="Contents", depth=2)

# ╔═╡ 5e7f4003-8067-5ddf-97b1-b4ade7c4378f
begin
	# Slots 1–6 of the validated categorical palette, in fixed order (as in your FittedModel notebooks).
	const state_colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
	const muted = "#52514e"
	default(; fontfamily="sans-serif", linewidth=2, framestyle=:axes, grid=true, gridalpha=0.12,
		tickfontcolor=muted, guidefontcolor=muted, foreground_color_axis=muted,
		foreground_color_border=muted, legend_foreground_color=:transparent,
		legend_background_color=:transparent, size=(720, 320), left_margin=3Plots.mm,
		bottom_margin=3Plots.mm)
	sig(x, d=3) = x isa Number && isfinite(x) ? round(x; sigdigits=d) : "–"
	pct(x, d=1) = x isa Number && isfinite(x) ? "$(round(100x; digits=d))%" : "–"
	mdtable(header, rows) = Markdown.parse(join(
		["| " * join(r, " | ") * " |" for r in [header, fill("---", length(header)), rows...]], "\n"))
	function table(dir, name)
		data, header = readdlm(joinpath(dir, name), ','; header=true)
		keys = Tuple(Symbol.(vec(header)))
		cell(x) = x isa Number || x isa Bool ? x : something(tryparse(Float64, string(x)), string(x))
		[NamedTuple{keys}(Tuple(cell.(data[i, :]))) for i in axes(data, 1)]
	end
	istrue(x) = x === true || x == "true"
	const results = joinpath(phase4, "results")
	suite = table(results, "suite.csv")
	suite_edges = table(results, "suite_edges.csv")
	find_fit(graph, noise, backend) = only(filter(r -> r.case == graph && r.noise == noise && r.backend == backend, suite))
end

# ╔═╡ 8ab66be9-3442-519a-b012-f15aa5124461
md"""
## 1. The two constructions

**Simplex.** One variable per state, and an equilibrium ``\xi_k`` on each coordinate axis:

```math
\dot x_j = s\, x_j \Big(1 - |x|^2 + \sum_i a_{ij} x_i^2\Big) + \varepsilon\,\dot W_j
```

``s`` is the speed you added. At ``\xi_k`` the eigenvalue in the ``x_j`` direction is ``s\,a_{kj}``, so ``a_{ij} > 0`` on an edge sets the escape rate and ``a_{ij} < 0`` elsewhere sets the contraction. Graphs need no 1- or 2-cycles.

**Cylinder.** One variable ``y_\ell`` per edge, and a line ``p`` on which vertices sit at the integers:

```math
\dot y_j = -s\, y_j\, G_j(y, p) + \varepsilon\,\dot W_j, \qquad
\dot p = s\,\Big(-\sin 2\pi p + \sum_j y_j^2 \tanh(p_{\omega j} - p)\Big) + \varepsilon\,\dot W_0
```

```math
G_j = \big(y_j^2 - \tfrac54\big)^2 - 1 - L_{\alpha j}\,\mathrm{sech}^2\big(k(p - p_{\alpha j})\big) + L_{\omega j}\,\mathrm{sech}^2\big(k(p - p_{\omega j})\big) + \sum_{i\ne j} y_i^2
```

The escape rate along edge ``j`` from its start vertex is ``s(L_{\alpha j} - \tfrac{9}{16} - f_{\omega j})``. Only 1-cycles are excluded, so a 2-cycle such as the worm's Forward ⇄ Pause 1 needs no hidden state here.

## 2. Checks against the paper

Before any fit, `precheck.jl` ran these against things the paper or algebra fixes.

| Check | Result |
|:--|:--|
| Horchler with a shared ``\alpha`` equals simplex at speed ``\alpha``, with expansion ``1/\nu`` and contraction 1 | ``\rho`` matches to rounding (below ``2\times 10^{-14}``) on all six graphs |
| Simplex, decision graph, noise ``10^{-5}``: log₁₀ of the branch proportions | ``-0.585, -0.596, -0.584``, against about ``-0.59`` read by eye from their Figure 12 |
| Same at noise ``10^{-8}`` | ``-0.789, -0.749, -0.771``, against about ``-0.80`` |
| Cylinder, Petersen graph, their parameters (``k_\omega = 0.4705``) | no real edge used, every transition off the graph |
| Cylinder, Petersen graph, ``k_\omega = 2.017`` | all 30 edges used, none off the graph |

With ``k_\omega = 0.4705`` the expanding eigenvalue for neighbouring vertices is ``-0.387``, so that edge can't leave its vertex. I can't tell whether the printed value is a misprint or I'm reading their equations differently. It's why this notebook fixes ``k = 2``.

Because of the first row, the simplex backend starts from exactly what Horchler already does, so what's new is a separate rate per edge.
"""

# ╔═╡ 3268a1e6-5f7a-5615-82fd-7ded471df04d
md"""
## 3. The test suite

Each fit is formula seed, up to 4 correction passes, then 504 CMA-ES evaluations only if those don't pass on held-out seeds, then a short polish. Acceptance is scored on 64 held-out runs of ``250\sum\tau`` at ``dt = 0.01/s``: each dwell within 5%, each branch inside its 95% interval, fewer than 1% of transitions off the graph.
"""

# ╔═╡ 11344098-901b-562e-98fc-4d2451adc853
let
	why(r) = join(filter(!isempty, [istrue(r.dwell_ok) ? "" : "dwell", r.off >= 0.01 ? "off-graph" : "", istrue(r.branches_ok) ? "" : "branch"]), ", ")
	cell(r) = istrue(r.pass) ? "pass" : "fail: " * why(r)
	rows = [[g; [cell(find_fit(g, ε, b)) for (ε, b) in ((1e-3, "simplex"), (1e-3, "cylinder"), (0.1, "simplex"), (0.1, "cylinder"))]] for g in first.(NAMED)]
	counts(ε, b) = count(r -> r.noise == ε && r.backend == b && istrue(r.pass), suite)
	push!(rows, ["**passed**"; ["**$(counts(ε, b)) of 6**" for (ε, b) in ((1e-3, "simplex"), (1e-3, "cylinder"), (0.1, "simplex"), (0.1, "cylinder"))]])
	mdtable(["Graph", "Simplex, 10⁻³", "Cylinder, 10⁻³", "Simplex, 0.1", "Cylinder, 0.1"], rows)
end

# ╔═╡ 19f3957b-c538-51cb-8a67-de0d521953c6
md"""
Every graph here was run with no hidden states, including the two that Horchler needs them for (the Δ-clique and the conflicting shared successor).

Details for every fit. "Largest branch miss" is the biggest gap between a measured and designed branch probability, on the first held-out set.
"""

# ╔═╡ 86fc0acf-5f18-5581-85ec-3b18a602e855
let
	miss(r) = (e = filter(x -> x.case == r.case && x.noise == r.noise && x.backend == r.backend, suite_edges); isempty(e) ? "–" : string(round(maximum(x -> abs(x.measured - x.designed), e); digits=3)))
	mdtable(["Graph", "Noise", "Backend", "Solved by", "Parameters", "Worst dwell error", "RMS", "Off-graph", "Largest branch miss", "Second set branches", "Accepted", "Evals", "Seconds"],
		[[r.case, r.noise, r.backend, r.stage, Int(r.dim), pct(r.worst_err), pct(r.rms, 2), pct(r.off, 2), miss(r), istrue(r.repeat_branches_ok) ? "ok" : "miss", istrue(r.pass) ? "pass" : "fail", Int(r.evals), round(Int, r.seconds)] for r in suite])
end

# ╔═╡ 1492c9e7-b2fa-525e-90ce-ee2c0869e942
md"""
**Reading it.** Of the 11 fits that fail, 9 fail on branches alone, and the misses are small: the largest gap anywhere is 0.038. The 95% interval is very tight at these run lengths, about ±0.01, and in 8 of those 9 both held-out sets miss, so these are small real biases and not chance.

The two others are different. Simplex on figure-eight at noise ``10^{-3}`` has 1.43% of transitions off the graph, above the 1% limit. **Cylinder on the worm ring at noise 0.1 is the only real failure**: Forward comes out 47% short and 25% of transitions are off the graph, and it hasn't been diagnosed.

Dwell times are within 5% in 23 of 24 fits, and the worst of those is 2.8%.

## 4. The worm at noise 0.1, all backends

Phase 3's rows come from `worm_compare.csv`, scored on 64 runs of 4000 s. The suite's rows are scored on 64 runs of ``250\sum\tau``, about 1985 s, with the same seed numbers.
"""

# ╔═╡ faa97e87-47a9-5a8f-b0cc-70a3821298cc
let
	worm = table(phase3results, "worm_compare.csv")
	rows = Any[[r.model, pct(r.rms, 2), pct(r.off, 2), istrue(r.pass) ? "pass" : "fail"] for r in worm]
	for b in ("simplex", "cylinder")
		r = find_fit("worm ring", 0.1, b)
		push!(rows, [b * " (" * string(Int(r.dim)) * " parameters)", pct(r.rms, 2), pct(r.off, 2), istrue(r.pass) ? "pass" : "fail"])
	end
	mdtable(["Model", "RMS log error", "Off-graph", "Accepted"], rows)
end

# ╔═╡ 8c5711fa-b11e-5137-a4c9-2a72b008fe51
md"""
## 5. Watch one

One fitted model, noise included, simulated live. For the simplex the upper panel is activity ``x_i^2`` per state. For the cylinder it's the position ``p`` along the line, with vertices at the integers. The lower panel is the decoded state in both.
"""

# ╔═╡ 599dfb81-7b6f-5d6e-81c9-c075d82b2f99
@bind graph_choice Select([i => NAMED[i][1] for i in eachindex(NAMED)]; default=2)

# ╔═╡ 76e23e1d-5f97-5bbd-8a8b-1cf353f9ff54
@bind noise_choice Select([1 => "noise 0.001", 2 => "noise 0.1"]; default=2)

# ╔═╡ 28ec7586-90f3-5275-a834-c2fd58a3c91c
@bind backend_choice Select([1 => "simplex", 2 => "cylinder"]; default=1)

# ╔═╡ 2642d534-7fe2-5000-9065-f7cbe8e5926e
@bind watch_seed Select([1 => "seed 1", 2 => "seed 2", 3 => "seed 3"]; default=1)

# ╔═╡ 984c35e4-2fba-5fe0-afbf-b3ee099a2f13
watched = let
	name, P, τ = NAMED[graph_choice]
	ε₀ = (1e-3, 0.1)[noise_choice]
	backend = (:simplex, :cylinder)[backend_choice]
	spec = (; P, τ, form=:u, ε₀, sample=name == "worm ring" ? WORM_SPEC.sample : minimum(τ) / 10)
	file = joinpath(results, "theta_$(backend)_$(replace(name, r"[^A-Za-z0-9]+" => "_"))_$ε₀.csv")
	edges = edge_list(P)
	build = (backend === :simplex ? make_simplex(spec, edges) : make_cylinder(spec, edges)).build
	(; name, backend, spec, τ, m=build(vec(readdlm(file, ','))))
end

# ╔═╡ 6a95b1ad-c6e2-5846-b639-b78b13a3dc21
let
	(; name, backend, spec, τ, m) = watched
	T = 12sum(τ)
	every, dt = frame_steps(m, spec.sample, 0.02)
	n = size(spec.P, 1)
	names = name == "worm ring" ? WORM_STATES : ["state $i" for i in 1:n]
	if backend === :simplex
		sol = solve(remake(sde_problem(m; T); seed=UInt64(watch_seed)), EM(); dt, adaptive=false, saveat=spec.sample, maxiters=typemax(Int))
		labels = [argmax(abs.(u)) for u in sol.u]
		top = plot(; ylabel="activity x²", title="$name, simplex, noise $(spec.ε₀)", titlefontsize=11, legend=:outerright, xticks=false)
		for i in 1:n
			plot!(top, sol.t, [u[i]^2 for u in sol.u]; color=state_colors[i], lw=1.2, label=names[i])
		end
	else
		N = length(m.Lα)
		q = (; s=m.α[1], m.Lα, m.from, m.to, m.ε, m.visible, every, V=Frames())
		z0 = SVector{N + 1}(ntuple(k -> k <= N ? m.ε : 1.0, Val(N + 1)))
		prob = SDEProblem(drift_cylinder, noise_cylinder, z0, (0.0, T), q; seed=UInt64(watch_seed))
		sol = solve(prob, EM(); dt, adaptive=false, saveat=spec.sample, maxiters=typemax(Int))
		labels = Int[]
		held = 1
		for z in sol.u
			k = round(Int, z[end])
			if 1 <= k <= n && abs(z[end] - k) < 0.25 && maximum(abs, z[1:end-1]) < 0.5
				held = k
			end
			push!(labels, held)
		end
		top = plot(sol.t, [z[end] for z in sol.u]; color=state_colors[1], lw=1.2, label="", ylabel="position p",
			title="$name, cylinder, noise $(spec.ε₀)", titlefontsize=11, xticks=false, yticks=1:n)
	end
	bottom = plot(sol.t, labels; seriestype=:steppost, color=muted, lw=1.2, label="", xlabel="time (s)",
		yticks=(1:n, names), ylims=(0.5, n + 0.5))
	plot(top, bottom; layout=Plots.grid(2, 1; heights=[0.65, 0.35]), size=(760, 430))
end

# ╔═╡ 661e3d1a-ea69-5296-b6c8-479f261f9461
md"""
## 6. What Phase 4 hasn't measured

- **Dynamical class.** These fits are scored on dwell, branches and off-graph transitions only. Whether each one is heteroclinic, excitable or something else waits for the Phase 5 classifier.
- **Dwell variability.** Only means are fitted.
- **Real worm behaviour.** The worm ring here is the same Table 1 benchmark as in Phase 3, not the measured branching.
- **The real 2-cycle case.** No test graph has a 2-cycle, so the cylinder's one extra capability is untested.
"""

# ╔═╡ Cell order:
# ╠═14f808ff-1f8b-571a-9d67-f80a4deb04e2
# ╠═0b196ab5-3d19-53cc-aae0-1b31452f8e38
# ╠═6070479e-b106-539d-98b3-d9370310ded0
# ╠═5e7f4003-8067-5ddf-97b1-b4ade7c4378f
# ╠═8ab66be9-3442-519a-b012-f15aa5124461
# ╠═3268a1e6-5f7a-5615-82fd-7ded471df04d
# ╠═11344098-901b-562e-98fc-4d2451adc853
# ╠═19f3957b-c538-51cb-8a67-de0d521953c6
# ╠═86fc0acf-5f18-5581-85ec-3b18a602e855
# ╠═1492c9e7-b2fa-525e-90ce-ee2c0869e942
# ╠═faa97e87-47a9-5a8f-b0cc-70a3821298cc
# ╠═8c5711fa-b11e-5137-a4c9-2a72b008fe51
# ╠═599dfb81-7b6f-5d6e-81c9-c075d82b2f99
# ╠═76e23e1d-5f97-5bbd-8a8b-1cf353f9ff54
# ╠═28ec7586-90f3-5275-a834-c2fd58a3c91c
# ╠═2642d534-7fe2-5000-9065-f7cbe8e5926e
# ╠═984c35e4-2fba-5fe0-afbf-b3ee099a2f13
# ╠═6a95b1ad-c6e2-5846-b639-b78b13a3dc21
# ╠═661e3d1a-ea69-5296-b6c8-479f261f9461
