# Which phase of the engine's chain graph nests deepest, and by how many native frames per link:
# redefines Rocket's LazyObservable subscription and Subject emission with a probe that records the
# deepest backtrace seen, for chains of two lengths.
#   julia --project=<env> stack_frames.jl <variant> <outfile>

using ReactiveMP, Rocket, ExponentialFamily, BayesBase, StandardMessagePassingRules
import ReactiveMP: activate!, FactorNodeActivationOptions, RandomVariableActivationOptions, DataVariableActivationOptions, israndom, get_stream_of_marginals

const VARIANT = ARGS[1]
const OUT = ARGS[2]
const DEEPEST = Ref(0)
probe() = (DEEPEST[] = max(DEEPEST[], length(backtrace())); nothing)

# the probes: every subscription through a LazyObservable, every Subject emission, every
# materialisation of a deferred message
const PROBE_ON = Ref(true)
@eval Rocket function on_subscribe!(observable::LazyObservable{D}, actor) where {D}
    $(PROBE_ON)[] && $(probe)()
    stream = getstream(observable)
    if stream !== nothing
        return LazySubscription(subscribe!(stream, actor))
    else
        subscription = LazySubscription(observable)
        pushpending!(observable, subscription, actor)
        return subscription
    end
end
@eval ReactiveMP function as_message(message::DeferredMessage, cache::Nothing, messages, marginals)::Message
    $(probe)()
    computed = message.mappingFn(messages, marginals)
    setcache!(message, computed)
    return computed
end

bethe(interfaces) = (Tuple(n for (n, v) in interfaces if israndom(v)), Tuple((n,) for (n, v) in interfaces if !israndom(v))...)
mknode(fform, interfaces) = factornode(fform, interfaces, filter(!isempty, bethe(interfaces)))

function phases(n)
    x = [randomvar() for _ in 1:n]; y = [datavar() for _ in 1:n]; v = constvar(1.0)
    nodes = Any[mknode(NormalMeanVariance, [(:out, x[1]), (:μ, constvar(0.0)), (:v, constvar(10.0))])]
    push!(nodes, mknode(NormalMeanVariance, [(:out, y[1]), (:μ, x[1]), (:v, v)]))
    for i in 2:n
        push!(nodes, mknode(NormalMeanVariance, [(:out, x[i]), (:μ, x[i - 1]), (:v, v)]))
        push!(nodes, mknode(NormalMeanVariance, [(:out, y[i]), (:μ, x[i]), (:v, v)]))
    end
    out = Pair{String, Int}[]
    DEEPEST[] = 0
    foreach(v -> activate!(v, RandomVariableActivationOptions()), x)
    foreach(v -> activate!(v, DataVariableActivationOptions()), y)
    foreach(nd -> activate!(nd, FactorNodeActivationOptions()), nodes)
    push!(out, "activate" => DEEPEST[]); DEEPEST[] = 0
    got = Ref{Any}(nothing)
    subs = [subscribe!(get_stream_of_marginals(w), (m) -> (got[] = m)) for w in x]
    push!(out, "subscribe to marginals" => DEEPEST[]); DEEPEST[] = 0
    foreach(new_observation!, y, randn(n))
    push!(out, "observe (emission + materialisation)" => DEEPEST[]); DEEPEST[] = 0
    foreach(unsubscribe!, subs)
    return out
end

phases(20)
a = phases(100); b = phases(300)
open(OUT, "a") do io
    for ((k, da), (_, db)) in zip(a, b)
        line = join((VARIANT, "frames at deepest point: " * k, da, db, round((db - da) / 200; digits = 2)), '\t')
        println(line); println(io, line)
    end
end
