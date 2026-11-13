# The CTRNN arm again: plain CMA-ES stopped early, so test restarts and a DE-then-CMA hybrid.
include("optimizers.jl")
campaign(B, b_eval, b_holdout, ["random BBO DE", "random CES restarts", "random CES restarts fresh", "random DE then CES fresh"]; tag="_followup")
