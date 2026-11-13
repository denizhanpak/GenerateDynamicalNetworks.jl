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
	const phase5 = joinpath(@__DIR__, "..", "benchmarks", "phase5")
	include(joinpath(phase5, "ctrnn_design.jl"))   # the CTRNN edge design, both backends, and Phases 2 to 4 through them
end
#---
md"""
# GenerateDynamicalNetworks: Phase 5, classifier, two populations and CTRNN

Two more backends and the classifier that says what each fit actually is. The exit test from the plan: **the worm ring and foraging graphs meet tolerance, and known examples of each class are labelled correctly, including your two-node CTRNN fit.**

Choices you made for this phase:

| Choice | Setting |
|:--|:--|
| Classifier | noise-free runs from points along a noisy run. Ghost means periodic with peak/slowest speed at least 100. Starts that disagree are `unknown` |
| Two populations | one global speed, a signed ``\nu`` per edge, one shared noise, no hidden states |
| CTRNN | one cell per state, argmax decoder, all weights, biases and time constants, searched from three regimes. Added on 2026-11-10: an edge-by-edge design with no search |
| Test suite | two populations on the six graphs at noise ``10^{-3}`` and ``0.1``, CTRNN search on the worm ring and foraging at ``0.1``, CTRNN design on all six graphs at ``0.1`` |

The CTRNN also has one global gain, which the plan didn't list: the 2021 weights only make sense at gain ``1/\varepsilon = 20``.

The code is in `benchmarks/phase5/`. `backends.jl` holds both models, `classes.jl` the classifier, and `suite.jl` the fits this notebook reads. The control at the end simulates one fitted model live.
"""
#---
TableOfContents(title="Contents", depth=2)
#---
begin
	# Slots 1–6 of the validated categorical palette, in fixed order.
	const state_colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
	const muted = "#52514e"
	default(; fontfamily="sans-serif", linewidth=2, framestyle=:axes, grid=true, gridalpha=0.12,
		tickfontcolor=muted, guidefontcolor=muted, foreground_color_axis=muted,
		foreground_color_border=muted, legend_foreground_color=:transparent,
		legend_background_color=:transparent, size=(720, 320), left_margin=3Plots.mm,
		bottom_margin=3Plots.mm)
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
	const results = joinpath(phase5, "results")
	suite = table(results, "suite.csv")
	suite_edges = table(results, "suite_edges.csv")
	classes = table(results, "classes.csv")
	san(n) = replace(n, r"[^A-Za-z0-9]+" => "_")
	miss(r) = (e = filter(x -> x.case == r.case && x.noise == r.noise && x.backend == r.backend && x.regime == r.regime, suite_edges); isempty(e) ? "–" : string(round(maximum(x -> abs(x.measured - x.designed), e); digits=3)))
	why(r) = istrue(r.pass) ? "pass" : "fail: " * join(filter(!isempty, [istrue(r.dwell_ok) ? "" : "dwell", r.off >= 0.01 ? "off-graph" : "", istrue(r.branches_ok) ? "" : "branch"]), ", ")
end
#---
md"""
## 1. The two constructions

**Two populations** (Ashwin & Postlethwaite 2016, appendix A). One variable ``p_j`` per state and one ``y_k`` per edge:

```math
\dot p_j = s\big[p_j\big(F(1-|p|^2) + D(p_j^2|p|^2 - |p|^4)\big) + E(-Z_j^{o} + Z_j^{i})\big] + \varepsilon \dot W, \qquad
\dot y_k = -s\, y_k\big[(y_k^2-1)^2 + A - B_k p_{\alpha(k)}^2 + C(|y|^2 - y_k^2)\big] + \varepsilon \dot W
```

with ``Z_j^{o} = \sum_{\alpha(k)=j} y_k^2 p_j p_{\omega(k)}``, ``Z_j^{i} = \sum_{\omega(k)=j} y_k^2 p_{\alpha(k)}^2``, ``B_k = 1.5 - \nu_k``, and the paper's ``A=0.5, C=2, D=10, E=4, F=2``. ``\nu_k < 0`` puts edge ``k`` on the heteroclinic side and ``\nu_k > 0`` on the excitable side. The text extraction of their equation (39) lost a bracket, and this is my reading of it, which reproduces their scalings below.

**CTRNN.** ``\tau_i \dot y_i = -y_i + \sum_j w_{ij}\,\sigma(g(y_j + b_j))`` plus noise on ``y``. Searches start from the 2021 four-weight construction, from your ring oscillator ``(-5, -14)``, and from near the fold ``(-6.9, -15)``. In both backends the argmax decides the state.

**CTRNN edge design** (`ctrnn_design.jl`). While cell ``i`` is on, every other cell ``j`` is a one-dimensional noisy unit resting at ``w_{ji}`` below its threshold, so

```math
\tau_j \dot y_j = -y_j - d_{ij} + A_j\,\sigma(g y_j) + \tau_j \varepsilon \dot W .
```

The mean time for noise to fire it has an exact one-dimensional formula, so each edge ``i \to j`` gets one weight ``w_{ji} = -d_{ij}``, solved from its target rate ``p_{ij}/\bar\tau_i``. The rates of a state's successors add, which sets its dwell and its branch probabilities together. Self-weights ``A_j`` are 12 noise SDs, every other weight is ``-1.5 A_j`` so the cell that fires shuts the rest off, cell time constants are a quarter of the shortest dwell they take part in, and the gain makes the sigmoid's width ``1/g`` half a noise SD of the quietest cell. Correction passes then move each edge's time by its measured rate error, through the same formula. No search is used. A two-way edge ``i \leftrightarrow j`` needs a hidden relay cell, covered in section 7.

## 2. Checks against the papers

| Check | Result |
|:--|:--|
| Two populations at ``\nu = -0.01``: residence time against noise ``\eta_y = 10^{-2}, 10^{-3}, 10^{-4}`` | 89.7, 293.8, 523.1, so 229 per decade at low noise against a predicted 230 |
| Two populations at ``\nu = +0.01`` | switches at ``10^{-2}`` (residence 178), never at ``10^{-3}`` or below |
| 2021 CTRNN on a 4-ring, default weights | still without noise, on the graph at noise 0.05 and 0.1 (100%, 99.9%), about 33% on the graph from 0.15 upward |
| CTRNN speed | 85 ns per step in the static version against 178 in `CTRNNs.jl`, and the package gave identical results in serial and threaded runs |

## 3. The classifier
"""
#---
let
	rows = [[r.name, r.known, r.expected, r.class, r.expected == r.class ? "yes" : "**no**", r.outcomes] for r in filter(x -> x.group == "known", classes)]
	mdtable(["Model", "What your notes or theory say", "Expected", "Labelled", "Match", "The five runs"], rows)
end
#---
md"""
A number after `periodic` is the ratio of peak to slowest speed on the cycle. Two labels follow from the rule you agreed, not from your notes: your 2-node fit comes out as a ghost (ratio 188), and your haunted ring3 as a plain limit cycle (ratio 1.78). The ``(-6.9, -15)`` setting is `unknown` because one start settled on an equilibrium and the others found the coexisting cycle.

## 4. The test suite
"""
#---
let
	rows = [[r.case, r.noise, r.backend, r.regime, r.stage, Int(r.dim), pct(r.worst_err), pct(r.rms, 2), pct(r.off, 2), miss(r), why(r), Int(r.evals), round(Int, r.seconds)] for r in suite]
	mdtable(["Graph", "Noise", "Backend", "Start", "Solved by", "Parameters", "Worst dwell error", "RMS", "Off-graph", "Largest branch miss", "Result", "Evals", "Seconds"], rows)
end
#---
let
	two = filter(r -> r.backend == "two", suite)
	ct = filter(r -> r.backend == "ctrnn" && r.regime != "design", suite)
	de = filter(r -> r.backend == "ctrnn" && r.regime == "design", suite)
	best = [argmin(r -> r.gate_f, filter(x -> x.case == c, ct)) for c in ("worm ring", "foraging")]
	worst = isempty(de) ? nothing : argmax(r -> r.rms, de)
	Markdown.parse("""
	**Two populations pass $(count(r -> istrue(r.pass), two)) of $(length(two)).** Both failures are at noise 0.1 and on branches alone.

	**The CTRNN search passes $(count(r -> istrue(r.pass), ct)) of $(length(ct)).** Best held-out score by graph: worm ring $(best[1].regime) ($(round(best[1].gate_f; digits=2))), foraging $(best[2].regime) ($(round(best[2].gate_f; digits=2))). For comparison, the two-population fits score under 0.01.

	**The CTRNN edge design passes $(count(r -> istrue(r.pass), de)) of $(length(de)).** $(worst === nothing ? "" : "Its largest RMS dwell error is $(pct(worst.rms, 2)), on $(worst.case), and its largest off-graph share is $(pct(maximum(r -> r.off, de), 2)).")

	**Why the search failed.** Its correction passes assumed each state's dwell depends only on its own cell's time constant, with a slope of +1. `ctrnn_diagnose.jl` measured that response on the worm seeds and fits: it runs from about ``-1.7`` to ``+2`` and sits largely off the diagonal (`results/ctrnn_diagnose.log`). So those passes steered with the wrong map, and CMA-ES then had at most 504 evaluations in 25 to 36 dimensions. The diagnosis run was stopped by its time cap before the fold fit, so the fold fit's response wasn't measured.
	""")
end
#---
md"""
## 5. What every fit is

The classifier on all fits from Phases 4 and 5, with the noise-free runs started from points along a noisy run at each fit's noise level.
"""
#---
let
	fits = filter(x -> x.group == "fit", classes)
	kinds = ["heteroclinic", "excitable", "limit_cycle", "ghost", "unknown"]
	groupname(n) = first(split(n, " "))
	backends = unique(groupname.(getproperty.(fits, :name)))
	rows = [[b; [string(count(r -> groupname(r.name) == b && r.class == k, fits)) for k in kinds]] for b in backends]
	mdtable(["Backend", kinds...], rows)
end
#---
mdtable(["Fit", "Label", "The five runs"], [[r.name, r.class, r.outcomes] for r in filter(x -> x.group == "fit" && x.class != "heteroclinic", classes)])
#---
md"""
A `saddle` result means the noise-free run stalled at an unstable equilibrium. That shows the fit has saddles where it dwells, but one run can't confirm every connection in the network.

## 6. Watch one

One fit, noise included, simulated live. The upper panel is activity per state (``p_j^2`` or the firing rate), and the lower is the decoded state.
"""
#---
@bind fit_choice Select([i => "$(suite[i].backend) $(suite[i].regime), $(suite[i].case), noise $(suite[i].noise)" for i in eachindex(suite)]; default=8)
#---
@bind watch_seed Select([1 => "seed 1", 2 => "seed 2", 3 => "seed 3"]; default=1)
#---
let
	r = suite[fit_choice]
	_, P, τ = only(c for c in NAMED if c[1] == r.case)
	ε₀ = r.noise
	spec = (; P, τ, form=:u, ε₀, sample=r.case == "worm ring" ? WORM_SPEC.sample : minimum(τ) / 10)
	θ = vec(readdlm(joinpath(results, "theta_$(r.backend)_$(r.regime)_$(san(r.case))_$ε₀.csv"), ','))
	m = r.backend == "two" ? two_signed(spec, edge_list(P), θ) : ctrnn_model(spec, θ; cap=Inf)
	T = 12sum(τ)
	every, dt = frame_steps(m, spec.sample, 0.02)
	n = size(P, 1)
	f, z0, q = r.backend == "two" ? (drift_two, two_start(m), two_params(m)) : (drift_ctrnn, ctrnn_start(m), ctrnn_params(m))
	sol = solve(SDEProblem(f, noise_flat, z0, (0.0, T), q; seed=UInt64(watch_seed)), EM(); dt, adaptive=false, saveat=spec.sample, maxiters=typemax(Int))
	activity = r.backend == "two" ? [[z[i]^2 for i in 1:n] for z in sol.u] : [[1 / (1 + exp(-m.g * (z[i] + m.b[i]))) for i in 1:n] for z in sol.u]
	names = r.case == "worm ring" ? WORM_STATES : ["state $i" for i in 1:n]
	top = plot(; ylabel="activity", title="$(r.backend) $(r.regime), $(r.case), noise $ε₀", titlefontsize=11, legend=:outerright, xticks=false)
	for i in 1:n
		plot!(top, sol.t, [a[i] for a in activity]; color=state_colors[i], lw=1.2, label=names[i])
	end
	bottom = plot(sol.t, [argmax(a) for a in activity]; seriestype=:steppost, color=muted, lw=1.2, label="", xlabel="time (s)",
		yticks=(1:n, names), ylims=(0.5, n + 0.5))
	plot(top, bottom; layout=Plots.grid(2, 1; heights=[0.65, 0.35]), size=(760, 430))
end
#---
md"""
## 7. Two-way edges and the real worm

One weight can't carry a two-way edge ``i \leftrightarrow j``: ``w_{ij}`` would have to let noise from ``j`` fire ``i`` and also let ``j`` shut ``i`` off. So for each pair, the edge ``j \to i`` goes through a hidden **relay cell** ``k``. While ``j`` is on, ``k`` rests just below threshold and noise fires it, with the same first-passage formula as any other edge. ``k`` then drives ``i`` on over ``j``'s inhibition and holds every other state off, and ``i`` shuts ``k`` off. The decoder reads state cells only, so the readout goes straight from ``j`` to ``i``. Graphs with no two-way edge get no relay.

The correction also changed. A relay's handoff takes a nearly fixed time, so part of a dwell no longer responds to the edge weights, and in a race between nearly deterministic clocks the branch shares depend on the total rate as well as on the rate ratios. Each pass now takes one Broyden step per state, on its log exit rate and log branch ratios, with the response between them learned from pass to pass: 10 passes, the last 3 with 192 runs. The six one-way suite fits above were made with the earlier per-edge rule, which `by_state=false` still reproduces.

Acceptance is as before: held-out sets 1 and 2 at the finer step, and set 1 again at half that step. A third set is shown for information only.
"""
#---
begin
	include(joinpath(phase5, "twoway_cases.jl"))
	twoway = table(results, "ctrnn_twoway.csv")
	twoway_extra = table(results, "ctrnn_twoway_extra.csv")
	twoway_states = table(results, "ctrnn_twoway_states.csv")
	twoway_branches = table(results, "ctrnn_twoway_branches.csv")
	twoway_cv = table(results, "ctrnn_twoway_cv.csv")
	worm_data = table(results, "worm_data.csv")
	cases_order = [c[1] for c in TWOWAY]
	sort!(twoway; by=r -> something(findfirst(==(r.case), cases_order), 99))
	ok(x) = istrue(x) ? "ok" : "**outside**"
end
#---
let
	rows = map(twoway) do r
		e = only(filter(x -> x.case == r.case, twoway_extra))
		L = design_layout(twoway_spec(r.case))
		b = filter(x -> x.case == r.case, twoway_branches)
		misses = isempty(b) ? "–" : string(round(maximum(x -> max(abs(x.set1 - x.designed), abs(x.set2 - x.designed), abs(x.half_step - x.designed)), b); digits=3))
		[r.case, "$(length(L.τc)) ($(length(L.rel)))", pct(r.worst_err), pct(r.rms, 2), pct(r.off, 2), misses,
		 istrue(r.pass) ? "pass" : "fail", istrue(e.third_pass) ? "pass" : "fail", e.class, "$(Int(e.relay_preceded)) of $(Int(e.relay_switches))"]
	end
	mdtable(["Graph", "Cells (relays)", "Worst dwell error, set 1", "RMS, set 1", "Off-graph, set 1", "Largest branch miss, any set", "All three checks", "Third set", "Class", "Relay switches with the relay on just before"], rows)
end
#---
mdtable(["Graph", "Edge", "Target", "Set 1", "Set 2", "Half step"],
	[[r.case, "$(Int(r.from))→$(Int(r.to))", round(r.designed; digits=3), "$(round(r.set1; digits=3)) $(ok(r.set1_ok))",
	  "$(round(r.set2; digits=3)) $(ok(r.set2_ok))", "$(round(r.half_step; digits=3)) $(ok(r.half_step_ok))"] for r in twoway_branches])
#---
md"""
### The real worm

`worm_data.jl` counts transitions between consecutive Viterbi-decoded bouts in Roberts et al.'s raw tracks (122 worms) and takes the mean bout per state, leaving out each worm's first and last bout. Every transition seen in the tracks is an edge, rare ones included, which gives four two-way pairs and so four relays. The table compares those targets with the fitted CTRNN on the three acceptance runs, and the bout variability of both.
"""
#---
let
	names = WORM_STATES
	d(i) = only(filter(x -> x.kind == "dwell" && Int(x.from) == i, worm_data))
	c(i) = only(filter(x -> x.kind == "cv" && Int(x.from) == i, worm_data))
	st = filter(x -> x.case == "worm data", twoway_states)
	cv = filter(x -> x.case == "worm data", twoway_cv)
	isempty(st) ? md"The worm-data fit hasn't been checked yet." :
	mdtable(["State", "Bouts in the tracks", "Mean bout, tracks (s)", "Model set 1", "Set 2", "Half step", "Bout CV, tracks", "Bout CV, model"],
		[[names[i], Int(d(i).count), round(d(i).value; digits=3), round(st[i].set1; digits=3), round(st[i].set2; digits=3), round(st[i].half_step; digits=3),
		  round(c(i).value; digits=2), round(cv[i].cv; digits=2)] for i in 1:4])
end
#---
let
	names = WORM_STATES
	tr = filter(x -> x.kind == "transition", worm_data)
	b = filter(x -> x.case == "worm data", twoway_branches)
	model(i, j, f) = (r = filter(x -> Int(x.from) == i && Int(x.to) == j, b); isempty(r) ? "–" : string(round(getproperty(only(r), f); digits=3)))
	mdtable(["Transition", "Count in the tracks", "Share, tracks", "Model set 1", "Set 2", "Half step"],
		[["$(names[Int(r.from)]) → $(names[Int(r.to)])", Int(r.count), round(r.value; digits=3),
		  model(Int(r.from), Int(r.to), :set1), model(Int(r.from), Int(r.to), :set2), model(Int(r.from), Int(r.to), :half_step)] for r in tr])
end
#---
md"""
### Watch a two-way fit

State cells in colour, relay cells dashed, then the decoded state. The seed control from section 6 applies here too.
"""
#---
@bind twoway_choice Select([r.case => r.case for r in twoway]; default=(any(r -> r.case == "worm data", twoway) ? "worm data" : first(twoway).case))
#---
let
	spec = twoway_spec(twoway_choice)
	L = design_layout(spec)
	m = ctrnn_model(spec, vec(readdlm(joinpath(results, "theta_ctrnn_twoway_$(san(twoway_choice))_0.1.csv"), ',')); cap=Inf)
	n, N = L.n, length(L.τc)
	T = min(12sum(spec.τ), 60.0)
	every, dt = frame_steps(m, spec.sample, 0.02)
	sol = solve(SDEProblem(drift_ctrnn, noise_flat, ctrnn_start(m), (0.0, T), ctrnn_params(m); seed=UInt64(watch_seed)), EM();
		dt, adaptive=false, saveat=spec.sample, maxiters=typemax(Int))
	act(i) = [1 / (1 + exp(-m.g * (y[i] + m.b[i]))) for y in sol.u]
	names = startswith(twoway_choice, "worm") ? WORM_STATES : ["state $i" for i in 1:n]
	top = plot(; ylabel="firing rate", title="CTRNN relay design, $(twoway_choice)", titlefontsize=11, legend=:outerright, xticks=false)
	for i in 1:n
		plot!(top, sol.t, act(i); color=state_colors[i], lw=1.2, label=names[i])
	end
	for (r, (j, i)) in enumerate(L.rel)
		plot!(top, sol.t, act(n + r); color=state_colors[i], lw=1.0, ls=:dash, label="relay $(names[j]) → $(names[i])")
	end
	bottom = plot(sol.t, [argmax(i -> y[i], 1:n) for y in sol.u]; seriestype=:steppost, color=muted, lw=1.2, label="", xlabel="time (s)",
		yticks=(1:n, names), ylims=(0.5, n + 0.5))
	plot(top, bottom; layout=Plots.grid(2, 1; heights=[0.65, 0.35]), size=(820, 450))
end
#---
md"""
## 8. What Phase 5 hasn't measured

- **Dwell variability.** Only means are fitted. On the real worm graph, Forward and Reverse bouts vary about as much as in the tracks, but both pauses are far too regular (table in section 7): their exits are nearly deterministic races.
- **Beyond first order.** The real worm fit matches the mean bouts and the transition shares, so its bout sequence is a first-order chain. Whether the tracks have longer memory wasn't checked.
- **Two strict misses.** The hub graph and the real worm graph each miss one branch interval on one held-out set, by under 0.02, while their other checks pass. Every hub fit made so far shows the same offset on that set's seeds.
- **Whether a better CTRNN search would succeed.** The design made the question less pressing, and it wasn't tested.
- **The rest of the plan.** Phases 6 and 7 are untouched, and the package still has no `src/`.
"""
