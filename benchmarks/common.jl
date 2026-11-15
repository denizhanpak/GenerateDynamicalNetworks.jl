# What the benchmark suite runs: each backend with the package's default options, on the cases the prototype
# tested it on (worm data left out: the package ships no real data), and the CSV helpers.
using GenerateDynamicalNetworks
using Dates, Printf

const RESULTS = joinpath(@__DIR__, "results")
const CHECKPOINTS = joinpath(RESULTS, "checkpoints")

const ONE_WAY = ["foraging", "worm ring", "figure-eight", "Δ-clique", "shared, solvable", "shared, conflicting"]
const TWO_WAY = ["flip-flop", "two-way and exit", "hub with two returns", "worm with return"]
const FIXTURES = [(name, ε) for ε in (1e-3, 0.1) for name in ONE_WAY]
const PILOT = "pilot 3-ring"

"""The specification of a case: an example network, or the pilot (3-ring, dwells 1, 3, 10, frames of 0.1)."""
case_spec(name, ε) = name == PILOT ? TransitionSpec(ring(3), [1.0, 3.0, 10.0]; noise=ε, sample=0.1) : example(name; noise=ε)

"""Each backend: its options, its cases, and whether its fit can stop at a deadline and resume from a checkpoint."""
const BACKENDS = [
    "horchler" => (backend=Horchler(), cases=FIXTURES, resumable=false),
    "simplex" => (backend=Simplex(), cases=FIXTURES, resumable=false),
    "cylinder" => (backend=Cylinder(), cases=FIXTURES, resumable=false),
    "twopop" => (backend=TwoPopulation(), cases=FIXTURES, resumable=false),
    "ctrnn" => (backend=CTRNNDesign(), cases=[(name, 0.1) for name in [ONE_WAY; TWO_WAY]], resumable=true),
    "synchrony" => (backend=Synchrony(), cases=[FIXTURES; [(PILOT, 1e-3), (PILOT, 0.1)]], resumable=true),
    "synchrony_race" => (backend=Synchrony(; seed=:race), cases=FIXTURES, resumable=true),
    "hopfield" => (backend=Hopfield(), cases=[FIXTURES; [(PILOT, 1e-3), (PILOT, 0.1)]], resumable=true),
]
const BACKEND_NAMES = first.(BACKENDS)
backend_entry(key) = last(only(b for b in BACKENDS if first(b) == key))

csvfield(x::AbstractString) = "\"" * replace(x, "\"" => "\"\"") * "\""
csvfield(x::Symbol) = String(x)
csvfield(x::AbstractFloat) = isnan(x) ? "NaN" : repr(x)
csvfield(x) = string(x)

"""Append one row to `file`, writing `header` first if the file is new."""
function append_row(file, header, row)
    mkpath(dirname(file))
    open(file, "a") do io
        filesize(file) == 0 && println(io, join(header, ","))
        println(io, join(csvfield.(row), ","))
    end
end

"""The rows of a CSV written by `append_row`, as vectors of strings (quoted fields may hold commas)."""
function read_rows(file)
    isfile(file) || return Vector{String}[]
    split_row(l) = [startswith(f, "\"") ? replace(f[2:end-1], "\"\"" => "\"") : String(f)
                    for f in split(l, r",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)")]
    return split_row.(readlines(file)[2:end])
end

case_key(backend, name, ε) = "$backend|$name|$ε"
file_key(backend, name, ε) = replace("$(backend)_$(name)_$ε", r"[^A-Za-z0-9.]+" => "_")
