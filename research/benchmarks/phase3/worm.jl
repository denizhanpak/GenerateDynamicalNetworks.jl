# The worm problem: the user's four-state cycle, targets and fitted model.
include(joinpath(@__DIR__, "optimize.jl"))

const WORM_STATES = ["Forward", "Pause 1", "Reverse", "Pause 2"]
const WORM_P = Float64[0 1 0 0; 0 0 1 0; 0 0 0 1; 1 0 0 0]
const WORM_SPEC = (; P=WORM_P, τ=[5.329, 0.441, 1.945, 0.208], form=:u, ε₀=0.1, sample=1 / 29.97)

# The user's own Horchler fit at noise 0.1 (from HorchlerFitNoise0.1.txt, not included): α per state, β = ν = 1.
const USER_FIT = let α = [0.5789603564403176, 7.486003342667031, 1.3852596336950647, 17.05699412601234], ν = ones(4), β = ones(4)
    (; form=:u, P=WORM_P, visible=4, κ=0.1, β, α, ν, ε=fill(0.1, 4), ρ=graph_rho(WORM_P, α, β, ν))
end
