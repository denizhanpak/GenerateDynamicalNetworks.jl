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

# ╔═╡ b76e9cc2-daed-11f1-be26-a36f994995b7
begin
	import Pkg
	Pkg.activate(@__DIR__)   # notebooks/Project.toml
	Pkg.instantiate()
	using Plots, PlutoUI, DelimitedFiles, Printf
	# The same packages horchler.jl loads. Pluto only re-imports what it sees in a cell, so they
	# are listed here too, or a cell rerun after a control moves can't find `solve` or `remake`.
	using SHCTools, StochasticDiffEq, StaticArrays, Graphs, Distributions, QuadGK, Roots
	using LinearAlgebra, Random, Statistics
	const phase2 = joinpath(@__DIR__, "..", "benchmarks", "phase2")
	include(joinpath(phase2, "horchler.jl"))    # the backend code: repair, design, simulate, measure
	include(joinpath(phase2, "cases.jl"))       # the test graphs
end

# ╔═╡ b77d46f0-daed-11f1-836b-e747e9eebb87
md"""
# GenerateDynamicalNetworks: Phase 2, the HorchlerLV backend

Phase 2 turns your SHCTools notebook's recipe into a backend and tests four things:

1. **Reproduction.** In the original form, does it give the same design and the same dwell table as `dwell_times_to_simulation.jl`?
2. **The u form.** Writing activity as ``x = u^2`` removes the clamp at zero. Does the design formula still hold?
3. **Repaired graphs.** Phase 1 repairs 2-cycles, Δ-cliques and conflicting branches with hidden states. Do the repaired networks run to specification?
4. **Shared successors.** Phase 1 replaced the greedy branch-noise rule with a linear solve. Does the result hold in simulation?

The code is one file, `benchmarks/phase2/horchler.jl`, which this notebook includes. Heavy runs are scripts whose results this notebook reads:

| File | What it holds |
|:-----|:--------------|
| `horchler.jl` | repair, branch noise, design for both forms, the StochasticDiffEq simulator, measurement |
| `cases.jl` | the named test graphs and Phase 1's random-graph generator |
| `reference.jl` | runs your SHCTools notebook as a script and saves its numbers to `reference/` |
| `sweeps.jl` | three passes per graph and form: first design, one correction, acceptance on held-out seeds at half the step |
"""

# ╔═╡ b77d48c6-daed-11f1-a315-ef23e80d77d6
TableOfContents(title="Contents", depth=2)

# ╔═╡ b77d4916-daed-11f1-8068-6d0359efeabd
begin
	const form_colors = Dict(:x => "#2a78d6", :u => "#eb6834")
	const state_colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#52514e"]
	const muted = "#52514e"
	default(; fontfamily="sans-serif", linewidth=2, framestyle=:axes, grid=true, gridalpha=0.12,
		tickfontcolor=muted, guidefontcolor=muted, foreground_color_axis=muted,
		foreground_color_border=muted, legend_foreground_color=:transparent,
		legend_background_color=:transparent, size=(720, 320), left_margin=3Plots.mm,
		bottom_margin=3Plots.mm)
	sig(x, d=3) = x isa Number && isfinite(x) ? round(x; sigdigits=d) : x
	pct(x, d=2) = x isa Number && isfinite(x) ? "$(x > 0 ? "+" : "")$(round(100x; digits=d - 1))%" : "–"
	mdtable(header, rows) = Markdown.parse(join(
		["| " * join(r, " | ") * " |" for r in [header, fill("---", length(header)), rows...]], "\n"))
	"""A results file as a vector of named rows, or `nothing` if it hasn't been run."""
	function table(path)
		isfile(path) || return nothing
		data, header = readdlm(path, ','; header=true)
		keys = Tuple(Symbol.(vec(header)))
		cell(x) = x isa Number || x isa Bool ? x : something(tryparse(Float64, string(x)), string(x))
		[NamedTuple{keys}(Tuple(cell.(data[i, :]))) for i in axes(data, 1)]
	end
	results(name) = table(joinpath(phase2, "results", name))
	istrue(x) = x === true || x == "true"
	formsym(x) = Symbol(string(x))
end

# ╔═╡ b77d4954-daed-11f1-a11d-07793d553518
md"""
## 1. Two forms of one network

The original form, as in Horchler et al. (2015) and SHCTools, works with activity directly:

```math
dx_i = x_i\Big(\alpha_i - \sum_j \rho_{ij} x_j\Big)dt + \varepsilon_i\,dW_i, \qquad x_i \ge 0 \text{ by clamping}
```

The u form, as in `NetworkModels.squared_glv_system`, works with its square root:

```math
du_i = u_i\Big(\alpha_i - \sum_j \rho_{ij} u_j^2\Big)dt + \varepsilon_i\,dW_i, \qquad x_i = u_i^2
```

The u form's field is odd in each ``u_i``, so ``u_i`` can cross zero and nothing needs clamping. Without noise, ``x = u^2`` obeys the original equation with every rate doubled. That fixes how the design carries over:

| | x form | u form |
|:--|:--|:--|
| escape rate from state ``k``, ``\lambda_u`` | ``\alpha_k/\nu_k`` | ``\alpha_k/\nu_k`` (the same) |
| neighbourhood edge, in the form's own variable | ``\kappa\beta`` | ``\sqrt{\kappa\beta}`` |
| transition between neighbourhoods | ``2\ln(1/\kappa - 1)/\alpha`` | ``\ln(1/\kappa - 1)/\alpha`` |
| noise acts on | ``x``, reflected at 0 | ``u``, free to cross 0 |

Everything else is shared: the branch race (it reads ``|u|``, which is the same half-normal race), the effective noise at a branch, one shared ``\alpha``, ``\nu`` per state, and ``\rho`` from `graph_rho`. `design(P, τ; form)` differs between the forms only in `edge_size` and `transition_time`.

The u form's edge is ``\sqrt{0.1} \approx 0.32`` rather than ``0.1``, so at the same noise its escape starts about 3 times further below the edge and its dwell depends less on the leading-order approximations.

### Watch one

Pick a test graph and a noise level. Both forms are designed for the same targets and run with the same seed. Grey dashed lines are hidden states.
"""

# ╔═╡ b77d498e-daed-11f1-a97b-7981a32069b0
@bind case_choice Select([c[1] => c[1] for c in NAMED]; default="foraging")

# ╔═╡ b77d49d4-daed-11f1-b19e-65f889fa7576
@bind noise_choice Select([1e-5 => "noise 1e-5", 1e-3 => "noise 1e-3", 1e-2 => "noise 1e-2"]; default=1e-5)

# ╔═╡ b77d49fc-daed-11f1-aefc-b31381e5e57b
chosen = only(filter(c -> c[1] == case_choice, NAMED))

# ╔═╡ b77d4a2e-daed-11f1-b089-5dbe3742eb71
"""Activity over time for plotting: one short run, saved on a grid."""
function trace(m; T, seed=1)
	prob = sde_problem(m; T)
	dt = 0.01 / maximum(m.α)
	sol = solve(remake(prob; p=(; prob.p..., V=Visits())), EM(); dt, adaptive=false, callback=READ,
		saveat=range(0, T, 1200), seed)
	return sol.t, reduce(hcat, [activity(m.form, u) for u in sol.u])
end

# ╔═╡ b77d4a60-daed-11f1-9aec-a3054fa51ce6
traces = map((:x, :u)) do form
	try
		m = design(chosen[2], chosen[3]; form, ε₀=noise_choice)
		t, X = trace(m; T=6 * sum(chosen[3]), seed=3)
		(; form, m, t, X, error="")
	catch err
		(; form, m=nothing, t=Float64[], X=zeros(0, 0), error=sprint(showerror, err))
	end
end

# ╔═╡ b77d4a92-daed-11f1-b036-ed85136fc99c
let
	panels = map(traces) do r
		p = plot(; ylabel="activity x", ylims=(-0.02, 1.1), legend=r.form === :x ? :outerright : false,
			title="$(r.form) form", titlefontsize=10, titlefontcolor=muted)
		if r.m === nothing
			annotate!(p, 0.5, 0.5, text("no design: " * first(r.error, 70), 8, muted))
		else
			n = size(r.X, 1)
			for i in 1:n
				hidden = i > r.m.visible
				plot!(p, r.t, r.X[i, :]; label=hidden ? "hidden $(i)" : "state $i",
					color=hidden ? muted : state_colors[mod1(i, 6)], linestyle=hidden ? :dash : :solid,
					linewidth=hidden ? 1.2 : 1.8)
			end
		end
		p
	end
	plot(panels...; layout=(2, 1), size=(720, 460), xlabel="time")
end

# ╔═╡ b77d4ad8-daed-11f1-ae21-617ac9dd8ec5
let rows = [[string(r.form), r.m === nothing ? "–" : sig(r.m.α[1]), r.m === nothing ? "–" : join(sig.(r.m.ν, 3), ", "),
			r.m === nothing ? r.error[1:min(end, 80)] : (isempty(r.m.reasons) ? "none" : join(r.m.reasons, "; "))]
			for r in traces]
	mdtable(["form", "shared α", "ν per state (hidden last)", "repairs"], rows)
end

# ╔═╡ b77d4b00-daed-11f1-bd20-47c70be831d5
md"""
## 2. Reproducing your notebook

`reference.jl` ran `dwell_times_to_simulation.jl` unchanged at its default settings: the foraging loop, noise ``10^{-5}``. In the x form the backend's design is the same to rounding:
"""

# ╔═╡ b77d4b28-daed-11f1-bb9e-73690ec24bd4
reference = (design=table(joinpath(phase2, "reference", "design.csv")), dwell=table(joinpath(phase2, "reference", "table.csv")),
	counts=readdlm(joinpath(phase2, "reference", "counts_first.csv"), ','))

# ╔═╡ b77d4b64-daed-11f1-9c22-f3d7ac05da07
let
	P, τ = NAMED[1][2], NAMED[1][3]
	m = design(P, τ; form=:x, ε₀=1e-5)
	rel(a, b) = maximum(abs.(a ./ b .- 1))
	d = reference.design
	mdtable(["quantity", "largest relative difference from SHCTools"],
		[["α", sig(rel(m.α, [r.alpha for r in d]), 2)],
		 ["ν", sig(rel(m.ν, [r.nu for r in d]), 2)],
		 ["ε", sig(rel(m.ε, [r.eps for r in d]), 2)],
		 ["effective ε", sig(rel(m.ε_eff, [r.eps_eff for r in d]), 2)]])
end

# ╔═╡ b77d4b96-daed-11f1-8b93-613b0d7a1771
md"""
The inversion is done differently, by root finding on `passagetime` instead of `lv_params`' closed form, so that both forms can share it. On three test points it agreed with `lv_params` to every printed digit (α = 26.974936, 133.624182 and 81.147901).

The dwell table follows. Your notebook ran one seed for 3000 time units and entered a state at ``(1-\kappa)\beta = 0.9``. The sweep ran 16 seeds for 600 each, using the Phase 1 decoder: enter at 0.5 and hold until another state crosses. In both, a correction pass requests ``\tau + (\tau - \text{measured})``.
"""

# ╔═╡ b77d4bc8-daed-11f1-9165-edd9e50f3a85
named = results("named_1e-5_states.csv")

# ╔═╡ b77d4bf2-daed-11f1-928d-b51d502ec9bc
function dwell_of(rows, case, form, pass)
	r = filter(r -> r.case == case && formsym(r.form) == form && r.pass == pass, rows)
	sort!(r; by=x -> x.state)
	return (dwell=[x.dwell for x in r], sem=[x.sem for x in r], target=[x.target for x in r], visits=[x.visits for x in r])
end

# ╔═╡ b77d4c24-daed-11f1-a2b3-d9e37a78a450
let
	ref = reference.dwell
	xf, xc = dwell_of(named, "foraging", :x, "first"), dwell_of(named, "foraging", :x, "corrected")
	mdtable(["state", "target", "SHCTools, first", "x form, first", "SHCTools, corrected", "x form, corrected"],
		[[r.state, r.target, "$(sig(r.first, 4)) ($(pct(r.first / r.target - 1)))", "$(sig(xf.dwell[i], 4)) ($(pct(xf.dwell[i] / r.target - 1)))",
		  "$(sig(r.corrected, 4)) ($(pct(r.corrected / r.target - 1)))", "$(sig(xc.dwell[i], 4)) ($(pct(xc.dwell[i] / r.target - 1)))"]
		 for (i, r) in enumerate(ref)])
end

# ╔═╡ b77d4c5e-daed-11f1-bab4-ed89299dae7c
md"""
Both rounds land within a point or two of your notebook's, and the corrected pass is within 1% in both. The largest gap is `left` on the first pass: −3.3% here against −5.4%. That's about 4 of your run's standard errors, and `left` is the shortest state, which is where the two decoders differ most. Phase 1 found the same thing: entry thresholds move short dwells through differences in rise time. Branch fractions agree too: `left` took $(sig(reference.counts[2, 3] / sum(reference.counts[2, :]), 3)) of departures from `scan` in your run against a target of 0.7.

`graph_rho` reproduces `createcycle` on a 4-cycle (checked when the code was written), so the exit test of the plan passes.
"""

# ╔═╡ b77d4c84-daed-11f1-9d26-939ec1157760
md"""
## 3. The u form against the x form

Every named graph at noise ``10^{-5}``, both forms, three passes. A pass meets the plan's acceptance when every dwell is within 5%, every branch is inside its 95% interval, and under 1% of transitions leave the graph.
"""

# ╔═╡ b77d4cb6-daed-11f1-b923-59cb7feb300d
named_cases = results("named_1e-5_cases.csv")

# ╔═╡ b77d4ce8-daed-11f1-bf21-356a60656c7a
"""Largest |error| of a case's dwells, from the per-state rows."""
worst_error(rows, case, form, pass) = let d = dwell_of(rows, case, form, pass)
	isempty(d.dwell) ? NaN : maximum(abs.(d.dwell ./ d.target .- 1))
end

# ╔═╡ b77d4d12-daed-11f1-aa92-519726db624e
fig_forms = let
	names = [c[1] for c in NAMED]
	passes = [("first", "first design"), ("accept", "corrected, held-out seeds, half step")]
	panels = map(passes) do (pass, label)
		p = plot(; title=label, titlefontsize=10, titlefontcolor=muted, ylabel="worst dwell error",
			ylims=(0, 19), yticks=(0:5:15, ["0", "5%", "10%", "15%"]), legend=pass == "first" ? :topright : false,
			xticks=(1:length(names), names), xrotation=20)
		for (k, form) in enumerate((:x, :u))
			w = [100worst_error(named, c, form, pass) for c in names]
			bar!(p, (1:length(names)) .+ (k - 1.5) * 0.36, w; bar_width=0.34, color=form_colors[form],
				linewidth=0, label="$(form) form")
		end
		hline!(p, [5]; color=muted, linestyle=:dash, linewidth=1, label="")
		p
	end
	plot(panels...; layout=(1, 2), size=(760, 330), bottom_margin=8Plots.mm)
end

# ╔═╡ b77d4d6c-daed-11f1-83bc-2be3aa69517c
let
	names = [c[1] for c in NAMED]
	verdict_of(case, form, pass) = let r = filter(r -> r.case == case && formsym(r.form) == form && r.pass == pass, named_cases)
		isempty(r) ? "–" : istrue(only(r).accepted) ? "pass" : "fail"
	end
	mdtable(["graph", "x first", "x corrected", "x held-out", "u first", "u corrected", "u held-out"],
		[[c, (verdict_of(c, f, p) * " ($(pct(worst_error(named, c, f, p), 2)))" for f in (:x, :u) for p in ("first", "corrected", "accept"))...]
		 for c in names])
end

# ╔═╡ b77d4d9e-daed-11f1-bfcb-49bbb5652586
md"""
**The u form's first design is better on every graph.** Its worst error is 0.9–7.9% against 2.6–17.3% for the x form. One correction brings five of the six graphs within 1% in both forms, and those results hold on held-out seeds at half the step.

**The worm ring separates them.** Forward is 25 times longer than Pause 2, and the x form's first design undershoots Forward by 17%. One correction leaves it at −5.5% on held-out seeds, just outside tolerance. The u form starts at −7.9% and finishes at −0.3%. Your SHCTools notebook already warned that wide dwell ratios need repeated corrections in the x form. The u form needs one here.

The likely reason is the size of the escape. In the u form the neighbourhood edge is ``\sqrt{0.1}``, not 0.1, so the leading-order passage time covers a longer stretch of the escape, where it's more accurate. Nothing above tests that explanation directly.
"""

# ╔═╡ b77d4dd0-daed-11f1-956c-679daf04ad33
md"""
## 4. Branches, including shared successors

Each row is a branch on the held-out pass. The interval is ``\pm 1.96\sqrt{p(1-p)/n}`` for ``n`` departures.
"""

# ╔═╡ b77d4e02-daed-11f1-849c-690750f59b67
named_branches = results("named_1e-5_branches.csv")

# ╔═╡ b77d4e48-daed-11f1-ac5c-814e3733b4fb
let rows = filter(r -> r.pass == "accept", named_branches)
	mdtable(["graph", "form", "branch", "designed", "measured", "departures", "inside 95%"],
		[[r.case, string(r.form), "$(Int(r.from)) → $(Int(r.to))", r.designed, sig(r.measured, 3), Int(r.departures),
		  istrue(r.within_ci) ? "yes" : "**no**"] for r in rows])
end

# ╔═╡ b77d4e7a-daed-11f1-90d3-ad348168269d
outside = let r = filter(r -> !istrue(r.within_ci), named_branches)
	(n=length(r), total=length(named_branches), rows=r)
end

# ╔═╡ b77d4eac-daed-11f1-b1e1-29bc6e1f8c2e
md"""
Across all three passes, $(outside.n) of $(outside.total) branch checks fell outside their 95% interval. At 95%, about $(round(Int, 0.05outside.total)) would be expected by chance. The miss was the u form's corrected pass on `shared, solvable` (0.625 against 0.6), and the same design was back inside its interval on the held-out pass.

**Shared successors work in both forms.**

- **`shared, solvable`** has state 3 following both branching states. The two noise ratios chain (``\varepsilon_2/\varepsilon_3`` and ``\varepsilon_3/\varepsilon_6``), so the linear solve finds one noise vector for both. Your notebook's greedy `branch_noise` rejects this graph, and here it runs with no hidden state.
- **`shared, conflicting`** has two states choosing between 2 and 3 at different odds, 70/30 and 40/60. No noise vector satisfies both. `repair` puts a hidden state on one shared edge, which gives that branch its own noise, and all four branches then land inside their intervals.
"""

# ╔═╡ b77d4ede-daed-11f1-84ce-b12a171ceda2
md"""
## 5. What repair builds

Two named graphs need hidden states. Each hidden state takes half its source's dwell, and the source gives up that half weighted by how often it uses the edge, so the source's mean dwell to the next visible state is unchanged:
"""

# ╔═╡ b77d4f1a-daed-11f1-9ab4-2724e10807c9
let
	rows = Any[]
	for (name, P, τ) in NAMED
		r = repair(P, τ)
		for (k, reason) in enumerate(r.reasons)
			h = r.visible + k
			from = findfirst(>(0), r.P[:, h])
			to = findfirst(>(0), r.P[h, :])
			push!(rows, [name, reason, "$from → hidden → $to", sig(r.τ[from], 3), sig(r.τ[h], 3), sig(τ[from], 3)])
		end
	end
	mdtable(["graph", "why", "new path", "source requests", "hidden requests", "source target"], rows)
end

# ╔═╡ b77d4f4c-daed-11f1-83a5-c12be710ea00
md"""
Hidden states don't show up in the dwell table, because the decoder only reads visible states. Time in a hidden state counts toward the state before it. The Δ-clique and `shared, conflicting` rows in section 3 are therefore the test that repair works: both pass in both forms after one correction, with no transitions off the graph.

### Random repaired graphs

Phase 1's generator, 6 states, edge density 0.3, uneven two-way branches, dwell drawn from 0.5–3. These need hidden states more often than the named graphs.
"""

# ╔═╡ b77d4f7e-daed-11f1-a503-c5c21177fa1c
random_cases = results("random_cases.csv")

# ╔═╡ b77d4fb0-daed-11f1-8d20-4966fca7e7e3
if random_cases === nothing
	md"""
	_Not run yet. It needs `sweeps.jl random`, about 20 minutes on 4 threads while another job is running, or about 7 on all 16 when the machine is free._
	"""
else
	let s = results("random_states.csv")
		graphs = unique([r.case for r in random_cases])
		mdtable(["form", "pass", "accepted", "median worst error", "hidden states (median)"],
			[[string(f), p,
			  "$(count(r -> formsym(r.form) == f && r.pass == p && istrue(r.accepted), random_cases)) of $(length(graphs))",
			  pct(median(filter(isfinite, [worst_error(s, g, f, p) for g in graphs])), 2),
			  sig(median([r.hidden for r in random_cases if formsym(r.form) == f && r.pass == p]), 2)]
			 for f in (:x, :u) for p in ("first", "corrected", "accept")])
	end
end

# ╔═╡ b77d4fe2-daed-11f1-8c28-271b05b8f3b4
md"""
## 6. Higher noise

The runs above are at ``10^{-5}``, where both forms are well inside their approximations. At ``10^{-3}`` and ``10^{-2}`` branches start to interfere and the leading-order passage time breaks down.
"""

# ╔═╡ b77d500a-daed-11f1-9c8f-e7a953f7b36b
noisy = results("named_noisy_cases.csv")

# ╔═╡ b77d503c-daed-11f1-adc8-0f3acc05dd43
if noisy === nothing
	md"""
	_Not run yet. It needs `sweeps.jl named 1e-3,1e-2 named_noisy`, about 15 minutes on 4 threads while another job is running._
	"""
else
	let s = results("named_noisy_states.csv")
		rows = Any[]
		for c in unique([r.case for r in noisy]), e in (1e-3, 1e-2)
			cells = map(Iterators.product(("first", "accept"), (:x, :u))) do (p, f)
				r = filter(r -> r.case == c && r.noise ≈ e && formsym(r.form) == f && r.pass == p, noisy)
				nd = filter(r -> r.case == c && r.noise ≈ e && formsym(r.form) == f && r.pass == "no design", noisy)
				!isempty(nd) ? "no design" : isempty(r) ? "–" :
					(istrue(only(r).accepted) ? "pass" : "fail") * " ($(pct(worst_error(filter(x -> x.noise ≈ e, s), c, f, p), 2)))"
			end
			push!(rows, [c, e, cells[1, 1], cells[2, 1], cells[1, 2], cells[2, 2]])
		end
		mdtable(["graph", "noise", "x first", "x held-out", "u first", "u held-out"], rows)
	end
end

# ╔═╡ b77d5078-daed-11f1-b812-5b8349bb10ad
md"""
## 7. Speed

Not measured cleanly. Another 16-thread job (`evolve.jl`) was running throughout, and the per-step times in the result files vary by 3× between identical passes. In Phase 0, on a quiet machine, the x form with its clamp callback cost 91 ns per step and the u form's solver without any callback 62. Both forms here use a callback to read visits, so most of that difference may be gone. This needs a rerun on a quiet machine.

## 8. Findings

| Question | Answer at noise ``10^{-5}`` |
|:---------|:----------------------------|
| Does the backend reproduce your notebook? | Yes. The design matches to rounding, and the dwell table agrees within a point or two on the first pass and within 1% after correction |
| Does the u form hold? | Yes, and it beats the x form on every first design. Only the u form passes the worm ring after one correction |
| Do repaired graphs run to specification? | Yes, for the two named graphs that need hidden states, in both forms |
| Does the linear branch-noise solve hold for shared successors? | Yes. A graph the greedy rule rejected runs without repair, and a conflicting one runs after repair |

**What isn't settled yet:**

- Higher noise and random repaired graphs (sections 5 and 6) haven't run.
- Speed hasn't been measured on a quiet machine.
- The reason the u form's first design is better is a hypothesis.
"""

# ╔═╡ Cell order:
# ╠═b76e9cc2-daed-11f1-be26-a36f994995b7
# ╟─b77d46f0-daed-11f1-836b-e747e9eebb87
# ╟─b77d48c6-daed-11f1-a315-ef23e80d77d6
# ╟─b77d4916-daed-11f1-8068-6d0359efeabd
# ╟─b77d4954-daed-11f1-a11d-07793d553518
# ╠═b77d498e-daed-11f1-a97b-7981a32069b0
# ╠═b77d49d4-daed-11f1-b19e-65f889fa7576
# ╟─b77d49fc-daed-11f1-aefc-b31381e5e57b
# ╟─b77d4a2e-daed-11f1-b089-5dbe3742eb71
# ╟─b77d4a60-daed-11f1-9aec-a3054fa51ce6
# ╟─b77d4a92-daed-11f1-b036-ed85136fc99c
# ╟─b77d4ad8-daed-11f1-ae21-617ac9dd8ec5
# ╟─b77d4b00-daed-11f1-bd20-47c70be831d5
# ╟─b77d4b28-daed-11f1-bb9e-73690ec24bd4
# ╟─b77d4b64-daed-11f1-9c22-f3d7ac05da07
# ╟─b77d4b96-daed-11f1-8b93-613b0d7a1771
# ╟─b77d4bc8-daed-11f1-9165-edd9e50f3a85
# ╟─b77d4bf2-daed-11f1-928d-b51d502ec9bc
# ╟─b77d4c24-daed-11f1-a2b3-d9e37a78a450
# ╟─b77d4c5e-daed-11f1-bab4-ed89299dae7c
# ╟─b77d4c84-daed-11f1-9d26-939ec1157760
# ╟─b77d4cb6-daed-11f1-b923-59cb7feb300d
# ╟─b77d4ce8-daed-11f1-bf21-356a60656c7a
# ╟─b77d4d12-daed-11f1-aa92-519726db624e
# ╟─b77d4d6c-daed-11f1-83bc-2be3aa69517c
# ╟─b77d4d9e-daed-11f1-bfcb-49bbb5652586
# ╟─b77d4dd0-daed-11f1-956c-679daf04ad33
# ╟─b77d4e02-daed-11f1-849c-690750f59b67
# ╟─b77d4e48-daed-11f1-ac5c-814e3733b4fb
# ╟─b77d4e7a-daed-11f1-90d3-ad348168269d
# ╟─b77d4eac-daed-11f1-b1e1-29bc6e1f8c2e
# ╟─b77d4ede-daed-11f1-84ce-b12a171ceda2
# ╟─b77d4f1a-daed-11f1-9ab4-2724e10807c9
# ╟─b77d4f4c-daed-11f1-83a5-c12be710ea00
# ╟─b77d4f7e-daed-11f1-a503-c5c21177fa1c
# ╟─b77d4fb0-daed-11f1-8d20-4966fca7e7e3
# ╟─b77d4fe2-daed-11f1-8c28-271b05b8f3b4
# ╟─b77d500a-daed-11f1-9c8f-e7a953f7b36b
# ╟─b77d503c-daed-11f1-adc8-0f3acc05dd43
# ╟─b77d5078-daed-11f1-b812-5b8349bb10ad
