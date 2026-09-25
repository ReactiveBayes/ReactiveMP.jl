# Micro and stream-side benchmarks for the Message{D} (A) vs untyped Message (B) question.
# Run with the variant's environment: julia --project=envA bench_micro.jl <tag> <outfile>
# Prints one line per case: name, min ns, median ns, allocs, bytes.

using ReactiveMP, Rocket, BayesBase, ExponentialFamily, Distributions, StandardMessagePassingRules, Statistics, LinearAlgebra
using MessagePassingRulesBase
using BenchmarkTools
import MessagePassingRulesBase: Target, ClusterTarget, DefaultAlgorithm
import ReactiveMP: MessageMapping, MarginalMapping, DeferredMessage, as_message, MessageProductContext, compute_product_of_messages,
    rule_arguments, MessageObservable, MarginalObservable, connect!, israndom, activate!, FactorNodeActivationOptions,
    RandomVariableActivationOptions, DataVariableActivationOptions, get_stream_of_marginals

const TAG = get(ARGS, 1, "?")
const OUT = get(ARGS, 2, "micro_results.tsv")
const VARIANT = let lazy = isdefined(ReactiveMP, Symbol("@invoke_callback"))
    ReactiveMP.Message isa UnionAll ? (isdefined(ReactiveMP, :new_message) ? "F" : lazy ? "D" : "A") : isdefined(ReactiveMP, :run_message_rule) ? (lazy ? "E" : "C") : "B"
end
const rows = String[]

function record(name, b)
    t = BenchmarkTools.minimum(b); m = BenchmarkTools.median(b)
    line = join((VARIANT, TAG, name, round(t.time; digits = 1), round(m.time; digits = 1), t.allocs, t.memory), '\t')
    println(line); push!(rows, line)
    return nothing
end
function record_raw(name, tmin, tmed, allocs, bytes)
    line = join((VARIANT, TAG, name, round(tmin; digits = 1), round(tmed; digits = 1), allocs, bytes), '\t')
    println(line); push!(rows, line)
    return nothing
end

bethe(interfaces) = (Tuple(n for (n, v) in interfaces if israndom(v)), Tuple((n,) for (n, v) in interfaces if !israndom(v))...)
mknode(fform, interfaces) = factornode(fform, interfaces, filter(!isempty, bethe(interfaces)))

msg(d) = Message(d, false, false)
mrg(d) = Marginal(d, false, false)

# The stream boundary: the engine hands a mapping values read from abstractly typed streams.
@noinline through_barrier(f, a::Ref{Any}, b::Ref{Any}) = f(a[], b[])

resolve_kernel(fform, target, mn, md, qn, qd) =
    MessagePassingRulesBase.find_message_rule(fform, target, DefaultAlgorithm(), ReactiveMP.kernel_arguments(mn, md, qn, qd))

cases = [
    # name, fform, target, msgs names, marginals names, messages, marginals
    (
        "NMV->out m[μ]::NMV m[v]::PM (BP)", NormalMeanVariance, Target{:out}(), Val((:μ, :v)), nothing,
        (msg(NormalMeanVariance(1.0, 2.0)), msg(PointMass(0.5))), nothing,
    ),
    (
        "NMP->μ q[out]::PM q[τ]::Gamma (VMP)", NormalMeanPrecision, Target{:μ}(), nothing, Val((:out, :τ)),
        nothing, (mrg(PointMass(0.3)), mrg(GammaShapeRate(2.0, 3.0))),
    ),
    (
        "MvNMC->out m[μ]::MvNMC(3) m[Σ]::PM (BP)", MvNormalMeanCovariance, Target{:out}(), Val((:μ, :Σ)), nothing,
        (msg(MvNormalMeanCovariance([1.0, 2.0, 3.0], Matrix(Diagonal([1.0, 2.0, 3.0])))), msg(PointMass(Matrix(1.0I, 3, 3)))), nothing,
    ),
    ("Bernoulli->out m[p]::Beta", Bernoulli, Target{:out}(), Val((:p,)), nothing, (msg(Beta(2.0, 3.0)),), nothing),
    ("Categorical->p q[out]::Categorical", Categorical, Target{:p}(), nothing, Val((:out,)), nothing, (mrg(Categorical([0.2, 0.3, 0.5])),)),
]

for (name, fform, target, mn, qn, ms, qs) in cases
    node = nothing
    mapping = MessageMapping(fform, target, mn, qn, DefaultAlgorithm(), nothing, node, nothing)
    r1 = mapping(ms, qs)
    # direct: concretely typed arguments, as after the stream's dynamic dispatch
    record("map/direct  " * name, @benchmark $mapping($ms, $qs))
    ra, rb = Ref{Any}(ms), Ref{Any}(qs)
    record("map/barrier " * name, @benchmark through_barrier($mapping, $ra, $rb))
    # rule_arguments + find_message_rule only
    # in C/E the engine resolves inside its kernel, over the unpacked data: time that path
    f = if VARIANT in ("C", "E")
        (m, q) -> resolve_kernel(fform, target, mn, ReactiveMP.data_of(m), qn, ReactiveMP.data_of(q))
    else
        (m, q) -> MessagePassingRulesBase.find_message_rule(fform, target, DefaultAlgorithm(), rule_arguments(mn, m, qn, q))
    end
    record("resolve/barrier " * name, @benchmark through_barrier($f, $ra, $rb))
    # deferred message materialisation
    g = (m, q) -> as_message(DeferredMessage(nothing, nothing, mapping), nothing, m, q)
    record("deferred/barrier " * name, @benchmark through_barrier($g, $ra, $rb))
end

# joint marginal (:out, :μ) of NMV, from two messages and q(v)
let
    mapping = MarginalMapping(NormalMeanVariance, ClusterTarget((:out, :μ)), Val((:out, :μ)), Val((:v,)), DefaultAlgorithm(), nothing)
    ms = (msg(NormalMeanVariance(1.0, 2.0)), msg(NormalMeanVariance(0.0, 1.0)))
    qs = (mrg(PointMass(0.5)),)
    ReactiveMP.compute_marginal(mapping, ms, qs)
    ra, rb = Ref{Any}(ms), Ref{Any}(qs)
    f = (m, q) -> ReactiveMP.compute_marginal(mapping, m, q)
    record("joint-marginal/barrier NMV (out,μ)", @benchmark through_barrier($f, $ra, $rb))
end

# products at a variable, over the Vector{AbstractMessage} collectLatest hands it
let
    v = randomvar()
    ctx = MessageProductContext()
    for (label, n) in (("3", 3), ("10", 10))
        msgs = AbstractMessage[msg(NormalMeanVariance(randn(), 1.0 + rand())) for _ in 1:n]
        record("product/NMV x$label", @benchmark compute_product_of_messages($v, $ctx, $msgs))
        dmsgs = AbstractMessage[(d = DeferredMessage(nothing, nothing, nothing); ReactiveMP.setcache!(d, m); d) for m in msgs]
        record("product/deferred NMV x$label", @benchmark compute_product_of_messages($v, $ctx, $dmsgs))
    end
    mvmsgs = AbstractMessage[msg(MvNormalMeanCovariance(randn(3), Matrix((1.0 + rand()) * I, 3, 3))) for _ in 1:3]
    record("product/MvNMC(3) x3", @benchmark compute_product_of_messages($v, $ctx, $mvmsgs))
    opts = RandomVariableActivationOptions()
    msgs = AbstractMessage[msg(NormalMeanVariance(randn(), 1.0 + rand())) for _ in 1:3]
    record("marginal-at-variable/NMV x3", @benchmark ReactiveMP._compute_marginal_from_messages($v, $opts, $msgs))
end

# the streams themselves: values pushed through MarginalObservable / MessageObservable and
# combineLatest(PushNew), as the engine wires them, to a sink
mutable struct Sink{T} <: Rocket.Actor{T}
    n::Int
end
Rocket.on_next!(s::Sink, _) = (s.n += 1; nothing)
Rocket.on_error!(::Sink, e) = throw(e)
Rocket.on_complete!(::Sink) = nothing

let
    sources = [Subject(Marginal) for _ in 1:3]
    obs = [MarginalObservable() for _ in 1:3]
    foreach((o, s) -> connect!(o, s), obs, sources)
    combined = combineLatest(Tuple(obs), PushNew()) |> map(Marginal, qs -> qs[1])
    sink = Sink{Marginal}(0)
    sub = subscribe!(combined, sink)
    vals = [mrg(NormalMeanVariance(randn(), 1.0)) for _ in 1:3]
    push3(sources, vals) = (
        for i in 1:3
            next!(sources[i], vals[i])
        end; nothing
    )
    record("stream/marginal combineLatest x3 round", @benchmark $push3($sources, $vals))
    unsubscribe!(sub)

    msources = [Subject(AbstractMessage) for _ in 1:3]
    mobs = [MessageObservable(AbstractMessage) for _ in 1:3]
    foreach((o, s) -> connect!(o, s), mobs, msources)
    mcombined = combineLatest(Tuple(mobs), PushNew()) |> map(AbstractMessage, ms -> ms[1])
    msink = Sink{AbstractMessage}(0)
    msub = subscribe!(mcombined, msink)
    mvals = [msg(NormalMeanVariance(randn(), 1.0)) for _ in 1:3]
    record("stream/message combineLatest x3 round", @benchmark $push3($msources, $mvals))
    unsubscribe!(msub)
end

# The callback event the mapping builds on every call, even with no callbacks: its `result`
# is inferred `Any` in every variant, so the event's type is instantiated at run time.
let
    mapping = MessageMapping(NormalMeanVariance, Target{:out}(), Val((:μ, :v)), nothing, DefaultAlgorithm(), nothing, nothing, nothing)
    ms = (msg(NormalMeanVariance(1.0, 2.0)), msg(PointMass(0.5)))
    rr = Ref{Any}(NormalMeanVariance(1.0, 3.0)); ann = ReactiveMP.AnnotationDict()
    ev(mapping, ms, rr, ann) = ReactiveMP.AfterMessageRuleCallEvent(mapping, ms, nothing, rr[], ann, nothing)
    record("event/AfterMessageRuleCallEvent, result::Any", @benchmark $ev($mapping, $ms, $rr, $ann))
end

# The cost of a function barrier alone: a kernel with one method, called with arguments whose
# types the caller does not know, versus the same call with known types.
@noinline kernel_one(data::Tuple, ann) = (data[1], ann)
@noinline kernel_two(a, b) = (a, b)
let
    data = (NormalMeanVariance(1.0, 2.0), PointMass(0.5)); ann = ReactiveMP.AnnotationDict()
    rd, ra = Ref{Any}(data), Ref{Any}(ann)
    record("barrier/kernel(tuple, ann), Any-typed args", @benchmark through_barrier(kernel_one, $rd, $ra))
    record("barrier/kernel(tuple, ann), known types", @benchmark kernel_one($data, $ann))
    r1, r2 = Ref{Any}(data[1]), Ref{Any}(data[2])
    record("barrier/kernel(a, b), Any-typed args", @benchmark through_barrier(kernel_two, $r1, $r2))
    # unpacking the data of two messages, as the mapping does before its barrier
    ms = (msg(NormalMeanVariance(1.0, 2.0)), msg(PointMass(0.5)))
    record("unpack/map(getdata, 2 messages)", @benchmark map(getdata, $ms))
end

# graph loops from scripts/benchmark_message_representation.jl
function iid_graph(n)
    x = randomvar(); y = [datavar() for _ in 1:n]; v = constvar(1.0)
    nodes = [mknode(NormalMeanVariance, [(:out, x), (:μ, constvar(0.0)), (:v, constvar(10.0))])]
    append!(nodes, [mknode(NormalMeanVariance, [(:out, y[i]), (:μ, x), (:v, v)]) for i in 1:n])
    return [x], y, nodes, [x]
end
function chain_graph(n)
    x = [randomvar() for _ in 1:n]; y = [datavar() for _ in 1:n]; v = constvar(1.0)
    nodes = Any[mknode(NormalMeanVariance, [(:out, x[1]), (:μ, constvar(0.0)), (:v, constvar(10.0))])]
    push!(nodes, mknode(NormalMeanVariance, [(:out, y[1]), (:μ, x[1]), (:v, v)]))
    for i in 2:n
        push!(nodes, mknode(NormalMeanVariance, [(:out, x[i]), (:μ, x[i - 1]), (:v, v)]))
        push!(nodes, mknode(NormalMeanVariance, [(:out, y[i]), (:μ, x[i]), (:v, v)]))
    end
    return x, y, nodes, x
end
function rungraph(graph, n, iterations, data)
    randoms, y, nodes, watched = graph(n)
    foreach(v -> activate!(v, RandomVariableActivationOptions()), randoms)
    foreach(v -> activate!(v, DataVariableActivationOptions()), y)
    foreach(nd -> activate!(nd, FactorNodeActivationOptions()), nodes)
    subs = [subscribe!(get_stream_of_marginals(w), (_) -> nothing) for w in watched]
    fe = subscribe!(bethe_free_energy(Float64, nodes, [randoms..., y...]), (_) -> nothing)
    stats = @timed for _ in 1:iterations
        foreach(new_observation!, y, data)
    end
    foreach(unsubscribe!, subs); unsubscribe!(fe)
    return stats
end
for (name, graph, n, it) in (("graph/iid n=1000 10 it", iid_graph, 1000, 10), ("graph/chain n=300 10 it", chain_graph, 300, 10))
    data = randn(n)
    rungraph(graph, n, it, data)
    samples = [rungraph(graph, n, it, data) for _ in 1:9]
    ts = map(s -> s.time * 1.0e9, samples)
    record_raw(name, minimum(ts), median(ts), -1, minimum(s -> s.bytes, samples))
end

open(OUT, "a") do io
    foreach(r -> println(io, r), rows)
end
