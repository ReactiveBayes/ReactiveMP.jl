# Rocket's hand-written combineLatest fast paths against simpler alternatives, one alternative per
# fresh process (the alternative is installed by redefining Rocket's storage constructors).
#   julia --project=<env> rocket_fastpaths.jl <alternative> <outfile>
# alternatives:
#   current   UInt8UpdatesStatus for <= 8 sources, MStorageN for <= 16 (as released)
#   bitarray  GenericUpdatesStatus (BitArrays + counters, P7) for every size, MStorageN
#   vecany    UInt8UpdatesStatus, Vector{Any} storage with `tuple(s...)` snapshots for every size
#   both      GenericUpdatesStatus and Vector{Any} storage
# Records, for k = 2, 3, 5, 8, 12, 16 sources: a steady PushNew round (ns, allocations) and the time
# of the first round for source types never seen before (compile time included, s).

using Rocket, BenchmarkTools, Statistics

const ALT = ARGS[1]
const OUT = ARGS[2]

if ALT in ("bitarray", "both")
    @eval Rocket getustorage(::Type{T}) where {T} = GenericUpdatesStatus(length(T.parameters))
end
if ALT in ("vecany", "both")
    @eval Rocket getmstorage(::Type{T}) where {T} = Vector{Any}(undef, length(T.parameters))
    @eval Rocket setstorage!(s::Vector{Any}, v, I::Int) = (s[I] = v)
end

mutable struct Box{I}
    v::Float64
end

struct Sink <: Rocket.Actor{Any} end
Rocket.on_next!(::Sink, _) = nothing
Rocket.on_error!(::Sink, e) = throw(e)
Rocket.on_complete!(::Sink) = nothing

pushall(sources, vals) = (foreach(next!, sources, vals); nothing)

function steady(k)
    sources = [Subject(Box) for _ in 1:k]
    sub = subscribe!(combineLatest(Tuple(sources), PushNew()), Sink())
    vals = [Box{i}(1.0) for i in 1:k]
    pushall(sources, vals)
    b = @benchmark $pushall($sources, $vals)
    unsubscribe!(sub)
    return minimum(b).time, minimum(b).allocs
end

# a fresh element type per trial, so each first round compiles the machinery for new types
function first_round(k, trial)
    T = Box{1000 * trial + k}
    sources = Tuple(Subject(Box{1000 * trial + 100 * k + i}) for i in 1:k)
    vals = [Box{1000 * trial + 100 * k + i}(1.0) for i in 1:k]
    t = @elapsed begin
        sub = subscribe!(combineLatest(sources, PushNew()), Sink())
        foreach(next!, sources, vals)
    end
    unsubscribe!(sub)
    return t
end

rows = String[]
for k in (2, 3, 5, 8, 12, 16)
    tsteady, allocs = steady(k)
    tfirst = median([first_round(k, trial) for trial in 1:5])
    line = join((ALT, k, round(tsteady; digits = 1), allocs, round(tfirst; digits = 4)), '\t')
    println(line); push!(rows, line)
end
open(io -> foreach(r -> println(io, r), rows), OUT, "a")
