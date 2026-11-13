# The interface between the core layer and the backends.

"""
    AbstractNetworkModel

A model a backend builds. A subtype `M` supplies these methods, and [`simulate`](@ref),
[`validate`](@ref), [`accept`](@ref) and [`classify`](@ref) then work on it:

| method | returns |
|:--|:--|
| `drift_function(m::M)` | the drift `f(u, q, t)` of `du = f(u, q, t) dt + g(u, q, t) dW` |
| `parameters(m::M)` | the parameter `NamedTuple` `q` passed to `f` and `g` |
| `initial_state(m::M)` | the starting state, an `SVector` |
| `speed(m::M)` | the fastest rate in the model; the step is at most `dtscale / speed(m)` |
| `frame_state(m::M, q, u)` | the state read at a frame: an index in `1:nstates`, or `0` to keep the last one |

and optionally

| method | default |
|:--|:--|
| `noise_function(m::M)` | `flat_noise`, the same amplitude `q.ε` on every variable |
| `in_bounds(m::M, u)` | `maximum(abs, u) < 100`; a run that leaves is stopped and marked failed |
| `project(m::M, u)` | `nothing`; otherwise the state is replaced by its return value after every step |
| `class_state(m::M, q, u)` | `frame_state(m, q, u)`; what [`classify`](@ref) reads, where values ≤ 0 are ignored |
| `class_start(m::M)` | `initial_state(m)`; where the classifier's noisy run starts |
"""
abstract type AbstractNetworkModel end

unimplemented(f, m) = throw(ArgumentError("$(typeof(m)) must implement GenerateDynamicalNetworks.$f; see AbstractNetworkModel"))

"""`drift_function(m)`: the drift `f(u, q, t)` of model `m`. See [`AbstractNetworkModel`](@ref)."""
drift_function(m::AbstractNetworkModel) = unimplemented(:drift_function, m)
"""`parameters(m)`: the parameters `q` of model `m`'s drift and noise. See [`AbstractNetworkModel`](@ref)."""
parameters(m::AbstractNetworkModel) = unimplemented(:parameters, m)
"""`initial_state(m)`: where runs of model `m` start. See [`AbstractNetworkModel`](@ref)."""
initial_state(m::AbstractNetworkModel) = unimplemented(:initial_state, m)
"""`speed(m)`: the fastest rate of model `m`, which sets the step. See [`AbstractNetworkModel`](@ref)."""
speed(m::AbstractNetworkModel) = unimplemented(:speed, m)
"""`frame_state(m, q, u)`: the state read at a frame, or 0 to keep the last. See [`AbstractNetworkModel`](@ref)."""
frame_state(m::AbstractNetworkModel, q, u) = unimplemented(:frame_state, m)

"""The same noise amplitude `q.ε` on every variable."""
flat_noise(u, q, t) = q.ε * ones(typeof(u))

noise_function(::AbstractNetworkModel) = flat_noise
in_bounds(::AbstractNetworkModel, u) = maximum(abs, u) < 100
project(::AbstractNetworkModel, u) = nothing
class_state(m::AbstractNetworkModel, q, u) = frame_state(m, q, u)
class_start(m::AbstractNetworkModel) = initial_state(m)
