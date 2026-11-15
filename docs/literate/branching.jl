# # A branching graph
#
# In a ring every state has one successor. This tutorial fits a graph with a choice in it, and looks at how
# the branch probabilities are measured and judged.

using GenerateDynamicalNetworks

# ## The specification
#
# The `"foraging"` example has five states. From state 2 the system goes to state 3 with probability 0.7 and to
# state 4 with probability 0.3; both lead to state 5 and back to the start.

spec = example("foraging"; noise=0.1)

#-

spec.P

#-

successors(spec, 2)

#-

spec.τ

# ## The fit
#
# This time the backend is [`Synchrony`](@ref), a network of phase oscillators. Nothing else changes: `fit`
# takes the specification and the backend.

model = fit(spec, Synchrony());

# ## One check
#
# [`validate`](@ref) reports the branches as well as the dwells.

check = validate(model, spec)

#-

check.branch

# Each entry is one branch: where it goes from and to, its designed probability, the share measured, and
# whether the measured share is inside the 95% interval around the designed one. The interval comes from the
# number of times the branching state was left:

n = sum(check.counts[2, :])

#-

halfwidth = 1.96sqrt(0.7 * 0.3 / n)

# So with this many visits a share passes only if it is within about 0.6 percentage points of 0.7.
#
# ## The protocol

acceptance = accept(model, spec)

# ## How much the measured share varies
#
# That interval is narrow, and it is the range in which a model with *exactly* the designed probability lands 95
# times in 100. So an exact model still fails about one check in twenty, and a model that is off by half a
# percentage point fails far more often ([How strict the branch rule is](@ref)).
#
# To see where this model stands, here are ten more sets of 64 runs, on seeds the protocol doesn't use.

starts = [9_400_001 + 200_000k for k in 0:9]
more = [validate(model, spec; seeds=s:s+63) for s in starts]
shares = [c.branch[1].measured for c in more]

# The number of these ten sets that fail the branch rule:

count(c -> !c.branch_ok, more)

# The figure shows twelve sets: the two of the protocol, then the ten more. Each bar is the 95% interval, drawn
# around the measured share. A set passes when its bar reaches the dashed line at 0.7. Blue points pass and red
# points fail.

using CairoMakie
CairoMakie.activate!(; type="png")

sets = [acceptance.set1; acceptance.set2; more]
measured = [c.branch[1].measured for c in sets]
half = [1.96sqrt(0.7 * 0.3 / sum(c.counts[2, :])) for c in sets]
passed = [c.branch_ok for c in sets]

fig = Figure(; size=(720, 280))
ax = Axis(fig[1, 1]; xlabel="set of runs", ylabel="share of 2 → 3", xticks=1:12)
hlines!(ax, 0.7; color=:black, linestyle=:dash)
errorbars!(ax, 1:12, measured, half; color=:gray40, whiskerwidth=8)
scatter!(ax, 1:12, measured; color=[p ? :steelblue : :firebrick for p in passed], markersize=10)
save("branching_shares.png", fig) #hide
nothing #hide

# ![The measured share of the branch from state 2 to state 3 in twelve sets of runs](branching_shares.png)
#
# The points don't scatter evenly around the line: nearly all are above it. The mean over the twelve sets and
# its standard error:

using Statistics
mean(measured), std(measured) / sqrt(length(measured))

# So this model sends the system to state 3 about half a percentage point too often. That is less than the
# half-width of the interval, which is why the model passed all three checks of the protocol, but it is enough
# for four of the ten further sets to fail.
#
# Two things follow. A model that passes has branch probabilities right to within about a percentage point at
# this run length, not exactly. And a model that fails one check may be no worse than one that passes: look at
# the measured shares, not only the verdict.
#
# The dwells are a different matter. Their rule is a fixed band of 5%, and the measured means vary much less
# than that from set to set:

[round.(100 .* c.err; digits=1) for c in more[1:3]]

# ## The dynamical class

classify(model, spec)
