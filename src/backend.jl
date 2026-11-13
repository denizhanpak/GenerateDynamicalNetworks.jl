# What every backend shares: the type a fitting recipe subtypes, `fit`, and the out-of-time signal
# that lets long fits run in batches.

"""
    AbstractBackend

A construction recipe. `fit(spec, backend)` builds a model of `spec` with it.
"""
abstract type AbstractBackend end

"""
    fit(spec::TransitionSpec, backend::AbstractBackend; kwargs...) -> AbstractNetworkModel

Build a model of `spec` with `backend`. See each backend for its options.
"""
function fit end

"""
    OutOfTime()

Thrown by a fit given a `deadline` (a `time()` value) instead of starting work it expects to finish
after it. With a `checkpoint` file the progress so far is saved, and calling `fit` again with the same
checkpoint resumes.
"""
struct OutOfTime <: Exception end
