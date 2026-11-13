### A Pluto.jl notebook ###
# v1.1.0

using Markdown
using InteractiveUtils

# ╔═╡ 0cb73ab2-daea-11f1-aa25-6191b44dc546
begin
	import Pkg
	Pkg.activate(@__DIR__)   # notebooks/Project.toml
	Pkg.instantiate()
	using Plots, PlutoUI, DelimitedFiles, Statistics, Printf
end

# ╔═╡ 0cc84fc8-daea-11f1-946d-2db4c06f4892
md"""
# GenerateDynamicalNetworks: Phase 0 benchmarks

Phase 0 chooses the simulator and the optimiser before any package code depends on them. Every package was installed in its own environment, `benchmarks/phase0/`, and its source was read before it was timed.

This notebook only reads results. The runs are the scripts beside them:

| Script | What it measures |
|:-------|:-----------------|
| `simulators.jl` | speed of four ways to simulate the worm ring, their agreement, thread scaling |
| `simulators2.jl` | path agreement, step size at three noise levels, network size |
| `gpukernel.jl` | whether DiffEqGPU's kernel can run the model on the CPU |
| `library_integrator.jl` | `StochasticDiffEq` set up like the loop: speed, agreement, ensemble threads |
| `optimizers.jl` | nine optimiser setups on two problems, judged on held-out seeds |
| `followup.jl` | the 24-parameter problem again, with restarts and a hybrid |

The machine is an AMD Ryzen 9 8945HS: 8 cores, 16 threads, and a Radeon 780M integrated GPU. There's no NVIDIA card, so CUDA ensembles were never an option.
"""

# ╔═╡ 0cc8518a-daea-11f1-b938-8b9f3678c048
TableOfContents(title="Contents", depth=2)

# ╔═╡ 0cc851da-daea-11f1-912c-f7b2a7036195
begin
	const state_colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
	const muted = "#52514e"
	default(; fontfamily="sans-serif", linewidth=2, framestyle=:axes, grid=true, gridalpha=0.12,
		tickfontcolor=muted, guidefontcolor=muted, foreground_color_axis=muted,
		foreground_color_border=muted, legend_foreground_color=:transparent,
		legend_background_color=:transparent, size=(720, 320), left_margin=3Plots.mm,
		bottom_margin=3Plots.mm, palette=state_colors)
	sig(x, d=3) = x isa Number && isfinite(x) ? round(x; sigdigits=d) : x
	mdtable(header, rows) = Markdown.parse(join(
		["| " * join(r, " | ") * " |" for r in [header, fill("---", length(header)), rows...]], "\n"))
	const results_dir = joinpath(@__DIR__, "..", "benchmarks", "phase0")
	"""A results file as a vector of named rows."""
	function table(name)
		data, header = readdlm(joinpath(results_dir, name), ','; header=true)
		keys = Tuple(Symbol.(vec(header)))
		cell(x) = x isa Number ? x : something(tryparse(Float64, string(x)), string(x))    # mixed columns stay text
		[NamedTuple{keys}(Tuple(cell.(data[i, :]))) for i in axes(data, 1)]
	end
	number(x) = x isa Number ? float(x) : something(tryparse(Float64, string(x)), NaN)
end

# ╔═╡ 0cc85216-daea-11f1-8302-5bfc6c080d9f
md"""
## 1. Simulator

The test is the worm ring from Phase 1 at noise ``10^{-3}``: 800 time units, about 7.3 million steps. Every simulator reads dwell with the same rule.

The first comparison set `StochasticDiffEq`'s defaults against a 30-line hand-written loop, `em_tally` in `common.jl`, which keeps the state in static arrays, clamps at zero and measures dwell as it goes. The loop won by 8×, but most of that gap was configuration. The second comparison (`library_integrator.jl`) sets the library up the same way, saving nothing and doing the clamp and dwell reading in one callback, and the package uses that.
"""

# ╔═╡ 0cc85252-daea-11f1-b310-edca52ce6f56
simulators = table("simulators.csv")

# ╔═╡ 0cc8528e-daea-11f1-9451-ebae130d4530
mdtable(["simulator", "seconds", "ns per step", "MB allocated", "speed vs lv_integrate"],
	[[r.name, sig(r.seconds), sig(r.ns_per_step), sig(r.mb), "$(sig(first(simulators).seconds / r.seconds, 2))×"]
	 for r in simulators])

# ╔═╡ 0cc852b6-daea-11f1-8b52-b55d1f013ab6
md"""
### The library, set up for search

Same seed for every row, so the dwell columns should match exactly.
"""

# ╔═╡ 0cc852de-daea-11f1-a399-35d5b3bc25ae
let s = table("library_integrator.csv"), base = first(s).ns_per_step
	mdtable(["setup", "ns per step", "MB allocated", "cost vs loop", "Forward", "Pause 1", "Reverse", "Pause 2"],
		[[r.name, sig(r.ns_per_step), sig(r.mb), "$(sig(r.ns_per_step / base, 2))×", (isnan(number(d)) ? "not read" : sig(number(d), 4) for d in (r.d1, r.d2, r.d3, r.d4))...]
		 for r in s])
end

# ╔═╡ 0cc852fc-daea-11f1-b410-053ccf1d0605
ensemble_text = let e = only(table("library_ensemble.csv"))
	string(sig(e.serial_s / e.threaded_s, 2), "× on ", Int(e.threads), " threads")
end

# ╔═╡ 0cc85324-daea-11f1-9513-0dcdbaae8c6d
md"""
The library reproduces the loop's dwell to every printed digit at about 4× the cost per step. About two thirds of that is the solver itself; the rest is the clamp callback, which also allocates 16 bytes a step. `EnsembleProblem` with `EnsembleThreads` gives $(ensemble_text) for 64 runs, against about 11× for the loop. The loop's speed isn't worth maintaining an integrator for, so it stays a benchmark reference only.

**Is it the same model?** With the same seed, the hand-written loop reproduces `lv_integrate` step for step: the largest difference over the first 20 steps is ``10^{-19}``, and the dwell agrees to six digits (`simulators2.jl`). Across 16 independent seeds per method, the means differ by at most 1.1 standard errors (`simulators.jl`).

**Ruled out by reading the source:**

- **`SimpleDiffEq.SimpleEM`** can't clamp, ignores callbacks, stores every step (about 256 MB for one of these runs) and uses the global random number generator.
- **`DiffEqGPU`'s `EnsembleGPUKernel` on the CPU** runs, but it silently ignored the clamp callback. Activity reached −0.315 in both the clamped and unclamped runs, so its output is a different model. It took 17.8 ns of wall time per step across 16 threads, roughly 200 ns of CPU time, against the loop's 21 on one thread.

### Threads

With the hand-written loop, independent runs scale well up to the 8 physical cores and gain a little beyond:
"""

# ╔═╡ 0cc8534c-daea-11f1-82ce-8928c5e247d6
let s = table("thread_scaling.csv"), base = first(s).seconds
	mdtable(["threads", "seconds", "speedup"], [[Int(r.threads), sig(r.seconds), "$(sig(base / r.seconds, 3))×"] for r in s])
end

# ╔═╡ 0cc85376-daea-11f1-bb81-714fe63c67bb
md"""
### Network size

Hidden states and oscillator backends make networks larger, so the hand-written loop was also timed on rings of more states. Static arrays stay ahead up to 32 states and compile slowly beyond that. The same split should apply to the library's out-of-place (static) and in-place (plain array) forms, but that hasn't been measured.
"""

# ╔═╡ 0cc8539c-daea-11f1-ac56-7df94ae05295
let s = table("network_size.csv")
	mdtable(["states", "static arrays, ns/step", "plain arrays, ns/step"],
		[[Int(r.N), isnan(number(r.static_ns)) ? "not run" : sig(number(r.static_ns)), sig(r.plain_ns)] for r in s])
end

# ╔═╡ 0cc853ce-daea-11f1-8c17-ebe295dfb105
md"""
## 2. Step size

A coarser step is the cheapest speed-up there is, so the question is how much it moves the dwell times. Each point is 256 runs, compared with the finest step, ``0.0025/\alpha``. Pause 2 is shown because it is the shortest state and moves the most.
"""

# ╔═╡ 0cc85408-daea-11f1-b5bb-a3ee5bdedffd
steps = table("step_size_noise.csv")

# ╔═╡ 0cc85432-daea-11f1-bc49-155cdd1273ce
fig_steps = let p = plot(; xlabel="step size (× 1/α)", ylabel="shift in Pause 2's dwell (%)", xscale=:log10,
		xticks=([0.01, 0.02, 0.04, 0.08, 0.16], ["0.01", "0.02", "0.04", "0.08", "0.16"]), xlims=(0.008, 0.25),
		legend=:outerright, size=(760, 320))
	hline!(p, [0]; color=muted, linewidth=1, label="")
	for (j, (e, name)) in enumerate(((1e-5, "1e-5"), (1e-3, "1e-3"), (1e-2, "1e-2")))
		rows = filter(r -> r.noise == e, steps)
		ref = first(rows)
		r = rows[2:end]
		y = [100 * (x.d4 / ref.d4 - 1) for x in r]
		err = [200 * sqrt(x.s4^2 + ref.s4^2) / ref.d4 for x in r]
		plot!(p, getfield.(r, :c), y; yerror=err, label="noise $name", color=state_colors[j], marker=:circle,
			markersize=4, markerstrokewidth=0)
		annotate!(p, 0.17, last(y), text("noise $name", 8, muted, :left))
	end
	p
end

# ╔═╡ 0cc8545a-daea-11f1-9a0d-e7c49ba76659
mdtable(["noise", "step (× 1/α)", "Forward", "Pause 1", "Reverse", "Pause 2", "seconds"],
	[let ref = first(filter(x -> x.noise == r.noise, steps))
		[r.noise, r.c, (string(sig(100 * (getfield(r, k) / getfield(ref, k) - 1), 2), "%") for k in (:d1, :d2, :d3, :d4))..., sig(r.seconds, 2)]
	end for r in steps])

# ╔═╡ 0cc85482-daea-11f1-8b43-33ddb733e443
md"""
The shift grows steadily with the step, and its size barely depends on the noise. Error bars are two standard errors. At ``0.04/\alpha`` the shortest state moves by 1.6–1.8%, four times faster than ``0.01/\alpha``. That's small against a 5% tolerance, but it's a bias, not noise, so it won't average out. Searching coarse and accepting fine handles it.

## 3. Optimiser

Two problems, chosen to pull in opposite directions:

| | Problem A: ring refit | Problem B: CTRNN |
|:--|:--|:--|
| Parameters | 5: shared α and four ν | 24: weights, biases, time constants |
| Start | the design formula, or a random point | a random point |
| What it tests | polishing a good guess | search from nothing on a rugged landscape |
| Budget | 1500 evaluations | 20000 evaluations |
| Repeats | 6 | 6, then 12 for the follow-up |

Every evaluation follows Phase 1: per-run mean dwell, averaged over runs. The final answer is judged on 48 seeds never used in search, at a finer step. A run **succeeds** if every state's mean dwell is within 5% and under 1% of transitions leave the graph.

"Fixed seeds" means every evaluation reuses the same noise. "Fresh" means new noise every time, with the answer taken as the population's mean instead of its luckiest point.
"""

# ╔═╡ 0cc854aa-daea-11f1-846b-6df304e4f2f2
begin
	runs_a = table("lv_ring_5d_runs.csv")
	runs_b = [table("ctrnn_24d_runs.csv"); [merge(r, (; method=r.method * " (×12)")) for r in table("ctrnn_24d_followup_runs.csv")]]
	methods(rows) = unique(getfield.(rows, :method))
	worst(r) = let w = number(r.worst_dwell_error); isfinite(w) ? 100w : 1000.0 end
	ok(r) = number(r.worst_dwell_error) <= 0.05 && number(r.off_graph) <= 0.01
end

# ╔═╡ 0cc854d2-daea-11f1-9096-ef15bc5200c7
function outcome_figure(rows; title)
	ms = methods(rows)
	p = plot(; xlabel="worst state's dwell error on held-out seeds (%)", xscale=:log10, title, titlefontsize=10,
		titlefontcolor=muted, yticks=(eachindex(ms), ms), yflip=true, legend=false, ylims=(0.4, length(ms) + 0.6),
		xticks=([1, 2, 5, 10, 20, 50, 100, 1000], ["1", "2", "5", "10", "20", "50", "100", "1000+"]),
		xlims=(0.5, 1500), size=(760, 60 + 34length(ms)), left_margin=12Plots.mm, right_margin=6Plots.mm)
	vline!(p, [5]; color=muted, linewidth=1, linestyle=:dash)
	for (k, m) in enumerate(ms)
		s = filter(r -> r.method == m, rows)
		x = clamp.(worst.(s), 0.5, 1000)
		y = k .+ range(-0.18, 0.18; length=length(s))
		scatter!(p, x, y; color=[ok(r) ? state_colors[1] : state_colors[2] for r in s], markersize=5, markerstrokewidth=0)
	end
	p
end

# ╔═╡ 0cc854f0-daea-11f1-b9c5-77dd3d399714
md"""
Each dot is one run. Blue runs succeeded and orange ones didn't; the dashed line is the 5% tolerance.
"""

# ╔═╡ 0cc85518-daea-11f1-bbf0-5394ad8dfff7
fig_outcomes_a = outcome_figure(runs_a; title="Problem A: worm ring refit, 5 parameters")

# ╔═╡ 0cc85536-daea-11f1-b289-0d525f5f9f74
summary(rows) = mdtable(["method", "runs", "succeeded", "median worst error", "median off-graph", "search − held-out gap"],
	[let s = filter(r -> r.method == m, rows)
		[m, length(s), count(ok, s), string(sig(median(worst.(s)), 2), "%"), string(sig(100median(number.(getfield.(s, :off_graph))), 2), "%"),
			sig(median(number.(getfield.(s, :heldout_f)) .- number.(getfield.(s, :search_f))), 2)]
	end for m in methods(rows)])

# ╔═╡ 0cc8555e-daea-11f1-8ed1-2f7c2cc2ac84
summary(runs_a)

# ╔═╡ 0cc85586-daea-11f1-95a0-e5acb8a51dbb
md"""
### How much does the design formula save?

The best value found so far on the search seeds, against evaluations used, as a median over the six runs:
"""

# ╔═╡ 0cc855ae-daea-11f1-8254-530704aace4d
fig_convergence = let tr = table("lv_ring_5d_trace.csv"), p = plot(; xlabel="evaluations", ylabel="best objective so far",
		xscale=:log10, yscale=:log10, legend=:outerright, size=(760, 320), ylims=(1e-11, 3),
		yticks=([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1], ["1e-10", "1e-8", "1e-6", "1e-4", "0.01", "1"]))
	hspan!(p, [1e-11, 1e-3]; color=muted, alpha=0.06, label="")
	annotate!(p, 1.2, 3e-11, text("below about 0.001 it is fitting the seeds' noise, not the dwell times", 8, muted, :left))
	for (j, m) in enumerate(("seeded CES", "random CES", "random BBO DE", "random SHADE"))
		grid = round.(Int, exp10.(range(0, log10(1500), 40)))
		curves = map(1:6) do rep
			t = filter(r -> r.method == m && r.rep == rep, tr)
			[let k = findlast(r -> r.evals <= n, t); k === nothing ? NaN : t[k].best end for n in grid]
		end
		med = [let v = filter(isfinite, getindex.(curves, i)); length(v) >= 4 ? median(v) : NaN end for i in eachindex(grid)]
		plot!(p, grid, med; label=m, color=state_colors[j])
	end
	p
end

# ╔═╡ 0cc855cc-daea-11f1-b88f-a778f1ae3b2f
md"""
From the formula, CMA-ES gets below 0.01 in about 60 evaluations, against about 170 from a random start. DE and SHADE need over 1,000, or never get there.

The curves are on the search seeds, so they show what fixed seeds do. CMA-ES keeps driving the objective down to about ``10^{-10}``, but the held-out errors in section 3 don't follow. Past roughly 0.001 it is fitting the particular noise in those 12 runs.

## 4. The 24-parameter problem
"""

# ╔═╡ 0cc855fe-daea-11f1-9206-b389c485060e
fig_outcomes_b = outcome_figure(runs_b; title="Problem B: 4-neuron CTRNN, 24 parameters")

# ╔═╡ 0cc85626-daea-11f1-81f0-69d43706f5c4
summary(runs_b)

# ╔═╡ 0cc8564e-daea-11f1-ab33-dd337cb69973
md"""
## 5. Decisions

| Question | Choice | Why |
|:---------|:-------|:----|
| Simulator | `StochasticDiffEq` `EM()`, nothing saved, clamp and dwell read in one callback | Same dwell as the loop on the same seed; about 4× slower per step, but no integrator to maintain. Full trajectories for plots come from the same solver |
| Parallelism | `EnsembleProblem` with `EnsembleThreads` | About 7× on 16 threads. `multi_threading=true` in CMA-ES gave 6.7× within a generation |
| Step size | Search at ``0.04/\alpha``, accept at ``0.01/\alpha`` or finer | 4× cheaper. The 1.6–1.8% bias is below tolerance and is caught at acceptance |
| Optimiser with a good start | `CMAEvolutionStrategy`, fresh noise, population mean | 6/6 successes on A; fixed seeds overfit, 3/6 |
| Optimiser from nothing | `BlackBoxOptim` DE, with independent repeats picked on held-out seeds | Best on B, but a quarter of runs still fail, so one run isn't enough |
| Dropped | Evolutionary, Metaheuristics, xNES, `SimpleEM`, `DiffEqGPU`, the hand-written loop | No advantage here, the wrong model, or code the library already provides |

### What changes in the plan

- **Fixed seeds were the wrong default.** The plan said search on common seeds and accept on held-out ones. On Problem A, fixed seeds drove the search objective to ``10^{-10}`` while held-out error stayed near 4.5%. That 4.5% is about what the sampling error of 12 short runs predicts (3.6% for Forward), so the optimiser was fitting the noise in those particular seeds. Fresh noise with the population mean fixed it.
- **The worm fit's 0.4% needs rechecking.** `horchler_dwell_fit` reports its objective on its own 64 search seeds. Those are much longer runs than here, so the overfit should be far smaller, but Phase 3's target should be a held-out number.
- **Restarts aren't optional.** Plain CMA-ES stopped early on Problem B whenever the population landed on a flat region, using a median of 12,122 of its 20,000 evaluations.

### What this doesn't show

- **Two problems is a small sample.** Problem A is a smooth five-parameter polish and Problem B one rugged landscape. Real backends may sit between them.
- **Repeats are few.** With 6 or 12 runs per method, differences of one or two successes don't mean much. The clear results are fixed against fresh noise on A and the restart failure on B.
- **The held-out step adds its own bias.** Acceptance runs at a finer step than search, which moves Pause 2 by about 0.9–1.3% (section 2). Part of the ~2% floor on Problem A is that bias, not the optimiser.
- **The optimiser timings used the hand-written loop.** With the library, every search costs about 4× more wall time per step. The comparisons between optimisers don't change, since all used the same simulator.
- **The CTRNN step size wasn't swept.** `dt = 0.005` was fixed, and its bias is unknown.
- **CMA-ES wasn't tried with fresh noise on Problem B from a good start.** There is no good start for a CTRNN yet. The Phase 5 four-weight construction might give one.
"""

# ╔═╡ Cell order:
# ╠═0cb73ab2-daea-11f1-aa25-6191b44dc546
# ╟─0cc84fc8-daea-11f1-946d-2db4c06f4892
# ╟─0cc8518a-daea-11f1-b938-8b9f3678c048
# ╟─0cc851da-daea-11f1-912c-f7b2a7036195
# ╟─0cc85216-daea-11f1-8302-5bfc6c080d9f
# ╟─0cc85252-daea-11f1-b310-edca52ce6f56
# ╟─0cc8528e-daea-11f1-9451-ebae130d4530
# ╟─0cc852b6-daea-11f1-8b52-b55d1f013ab6
# ╟─0cc852de-daea-11f1-a399-35d5b3bc25ae
# ╟─0cc852fc-daea-11f1-b410-053ccf1d0605
# ╟─0cc85324-daea-11f1-9513-0dcdbaae8c6d
# ╟─0cc8534c-daea-11f1-82ce-8928c5e247d6
# ╟─0cc85376-daea-11f1-bb81-714fe63c67bb
# ╟─0cc8539c-daea-11f1-ac56-7df94ae05295
# ╟─0cc853ce-daea-11f1-8c17-ebe295dfb105
# ╟─0cc85408-daea-11f1-b5bb-a3ee5bdedffd
# ╟─0cc85432-daea-11f1-bc49-155cdd1273ce
# ╟─0cc8545a-daea-11f1-9a0d-e7c49ba76659
# ╟─0cc85482-daea-11f1-8b43-33ddb733e443
# ╟─0cc854aa-daea-11f1-846b-6df304e4f2f2
# ╟─0cc854d2-daea-11f1-9096-ef15bc5200c7
# ╟─0cc854f0-daea-11f1-b9c5-77dd3d399714
# ╟─0cc85518-daea-11f1-bbf0-5394ad8dfff7
# ╟─0cc85536-daea-11f1-b289-0d525f5f9f74
# ╟─0cc8555e-daea-11f1-8ed1-2f7c2cc2ac84
# ╟─0cc85586-daea-11f1-95a0-e5acb8a51dbb
# ╟─0cc855ae-daea-11f1-8254-530704aace4d
# ╟─0cc855cc-daea-11f1-b88f-a778f1ae3b2f
# ╟─0cc855fe-daea-11f1-9206-b389c485060e
# ╟─0cc85626-daea-11f1-81f0-69d43706f5c4
# ╟─0cc8564e-daea-11f1-ab33-dd337cb69973
