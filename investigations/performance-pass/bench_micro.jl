# Micro, stream and engine-graph benchmarks, without RxInfer.
#   julia --project=<variant>/env bench_micro.jl <variant> <tag> <outfile>
# One TSV line per case: variant, tag, case, min ns, median ns, allocs, bytes.

using ReactiveMP, Rocket, BayesBase, ExponentialFamily, Distributions, StandardMessagePassingRules, Statistics, LinearAlgebra
using MessagePassingRulesBase
using BenchmarkTools
import MessagePassingRulesBase: Target, ClusterTarget, DefaultAlgorithm
import ReactiveMP: MessageMapping, MarginalMapping, DeferredMessage, as_message, MessageProductContext, compute_product_of_messages,
    rule_arguments, MessageObservable, MarginalObservable, connect!, israndom, activate!, FactorNodeActivationOptions,
    RandomVariableActivationOptions, DataVariableActivationOptions, get_stream_of_marginals

const VARIANT = get(ARGS, 1, "?")
const TAG = get(ARGS, 2, "?")
const OUT = get(ARGS, 3, "micro.tsv")
const ONLY = get(ENV, "BENCH_ONLY", "")
const rows = String[]

want(name) = isempty(ONLY) || any(o -> occursin(o, name), split(ONLY, ","))

function record(name, b)
    t = BenchmarkTools.minimum(b); m = BenchmarkTools.median(b)
    line = join((VARIANT, TAG, name, round(t.time; digits = 1), round(m.time; digits = 1), t.allocs, t.memory), '\t')
    println(line); push!(rows, line); open(io -> println(io, line), OUT, "a")
    return nothing
end
function record_raw(name, tmin, tmed, allocs, bytes)
    line = join((VARIANT, TAG, name, round(tmin; digits = 1), round(tmed; digits = 1), allocs, bytes), '\t')
    println(line); push!(rows, line); open(io -> println(io, line), OUT, "a")
    return nothing
end

bethe(interfaces) = (Tuple(n for (n, v) in interfaces if israndom(v)), Tuple((n,) for (n, v) in interfaces if !israndom(v))...)
mknode(fform, interfaces) = factornode(fform, interfaces, filter(!isempty, bethe(interfaces)))

msg(d) = Message(d, false, false)
mrg(d) = Marginal(d, false, false)

# The stream boundary: the engine hands a mapping values read from abstractly typed streams.
@noinline through_barrier(f, a::Ref{Any}, b::Ref{Any}) = f(a[], b[])

const cases = [
    (
        "NMV->out m[μ]::NMV m[v]::PM (BP)", NormalMeanVariance, Target{:out}(), Val((:μ, :v)), nothing,
        (msg(NormalMeanVariance(1.0, 2.0)), msg(PointMass(0.5))), nothing,
    ),
    (
        "NMP->μ q[out]::PM q[τ]::Gamma (VMP)", NormalMeanPrecision, Target{:μ}(), nothing, Val((:out, :τ)),
        nothing, (mrg(PointMass(0.3)), mrg(GammaShapeRate(2.0, 3.0))),
    ),
    (
        "NMP->τ q[out]::PM q[μ]::NMV (VMP)", NormalMeanPrecision, Target{:τ}(), nothing, Val((:out, :μ)),
        nothing, (mrg(PointMass(0.3)), mrg(NormalMeanVariance(0.0, 1.0))),
    ),
    (
        "MvNMC->out m[μ]::MvNMC(3) m[Σ]::PM (BP)", MvNormalMeanCovariance, Target{:out}(), Val((:μ, :Σ)), nothing,
        (msg(MvNormalMeanCovariance([1.0, 2.0, 3.0], Matrix(Diagonal([1.0, 2.0, 3.0])))), msg(PointMass(Matrix(1.0I, 3, 3)))), nothing,
    ),
    ("Bernoulli->out m[p]::Beta", Bernoulli, Target{:out}(), Val((:p,)), nothing, (msg(Beta(2.0, 3.0)),), nothing),
    ("Categorical->p q[out]::Categorical", Categorical, Target{:p}(), nothing, Val((:out,)), nothing, (mrg(Categorical([0.2, 0.3, 0.5])),)),
]

for (name, fform, target, mn, qn, ms, qs) in cases
    want("map") || want("resolve") || want("exec") || want("deferred") || continue
    mapping = MessageMapping(fform, target, mn, qn, DefaultAlgorithm(), nothing, nothing, nothing)
    mapping(ms, qs)
    record("map/direct  " * name, @benchmark $mapping($ms, $qs))
    ra, rb = Ref{Any}(ms), Ref{Any}(qs)
    record("map/barrier " * name, @benchmark through_barrier($mapping, $ra, $rb))
    # with a callbacks object that listens to nothing: what RxInfer's trace/benchmark modes pay
    cbmapping = MessageMapping(fform, target, mn, qn, DefaultAlgorithm(), nothing, nothing, (;))
    cbmapping(ms, qs)
    record("map/callbacks=(;) " * name, @benchmark $cbmapping($ms, $qs))
    # resolution alone
    f = (m, q) -> MessagePassingRulesBase.find_message_rule(fform, target, DefaultAlgorithm(), rule_arguments(mn, m, qn, q))
    record("resolve/barrier " * name, @benchmark through_barrier($f, $ra, $rb))
    # the rule body alone, the way the engine runs it
    args = rule_arguments(mn, ms, qn, qs)
    spec = ReactiveMP.resolve_rule(MessagePassingRulesBase.find_message_rule(fform, target, DefaultAlgorithm(), args))
    ctx = mapping.context
    ann = ReactiveMP.rule_annotations(mn, ms, qn, qs, ReactiveMP.AnnotationDict())
    alg = MessagePassingRulesBase.rule_algorithm(spec, DefaultAlgorithm())
    record("exec/execute_rule " * name, @benchmark MessagePassingRulesBase.execute_rule($spec, nothing, nothing, $alg, $ctx, $args, $ann, $target))
    # deferred message materialisation
    g = (m, q) -> as_message(DeferredMessage(nothing, nothing, mapping), nothing, m, q)
    record("deferred/barrier " * name, @benchmark through_barrier($g, $ra, $rb))
end

# joint marginal (:out, :μ) of NMV, from two messages and q(v)
if want("joint")
    mapping = MarginalMapping(NormalMeanVariance, ClusterTarget((:out, :μ)), Val((:out, :μ)), Val((:v,)), DefaultAlgorithm(), nothing)
    ms = (msg(NormalMeanVariance(1.0, 2.0)), msg(NormalMeanVariance(0.0, 1.0)))
    qs = (mrg(PointMass(0.5)),)
    ReactiveMP.compute_marginal(mapping, ms, qs)
    ra, rb = Ref{Any}(ms), Ref{Any}(qs)
    f = (m, q) -> ReactiveMP.compute_marginal(mapping, m, q)
    record("joint-marginal/barrier NMV (out,μ)", @benchmark through_barrier($f, $ra, $rb))
end

# products at a variable, over the Vector{AbstractMessage} collectLatest hands it
if want("product") || want("marginal-at")
    v = randomvar()
    ctx = MessageProductContext()
    for n in (2, 5, 10, 100)
        msgs = AbstractMessage[msg(NormalMeanVariance(randn(), 1.0 + rand())) for _ in 1:n]
        record("product/NMV x$n", @benchmark compute_product_of_messages($v, $ctx, $msgs))
    end
    msgs = AbstractMessage[msg(NormalMeanVariance(randn(), 1.0 + rand())) for _ in 1:10]
    dmsgs = AbstractMessage[(d = DeferredMessage(nothing, nothing, nothing); ReactiveMP.setcache!(d, m); d) for m in msgs]
    record("product/deferred NMV x10", @benchmark compute_product_of_messages($v, $ctx, $dmsgs))
    mvmsgs = AbstractMessage[msg(MvNormalMeanCovariance(randn(3), Matrix((1.0 + rand()) * I, 3, 3))) for _ in 1:3]
    record("product/MvNMC(3) x3", @benchmark compute_product_of_messages($v, $ctx, $mvmsgs))
    gmsgs = AbstractMessage[msg(GammaShapeRate(1.0 + rand(), 1.0 + rand())) for _ in 1:10]
    record("product/Gamma x10", @benchmark compute_product_of_messages($v, $ctx, $gmsgs))
    opts = RandomVariableActivationOptions()
    msgs3 = AbstractMessage[msg(NormalMeanVariance(randn(), 1.0 + rand())) for _ in 1:3]
    record("marginal-at-variable/NMV x3", @benchmark ReactiveMP._compute_marginal_from_messages($v, $opts, $msgs3))
end

# Rocket primitives in isolation
mutable struct Sink{T} <: Rocket.Actor{T}
    n::Int
end
Rocket.on_next!(s::Sink, _) = (s.n += 1; nothing)
Rocket.on_error!(::Sink, e) = throw(e)
Rocket.on_complete!(::Sink) = nothing

if want("rocket")
    # a subject's fan-out to k listeners
    for k in (1, 10, 100)
        s = Subject(AbstractMessage)
        subs = [subscribe!(s, Sink{AbstractMessage}(0)) for _ in 1:k]
        m = msg(NormalMeanVariance(0.0, 1.0))
        record("rocket/subject fan-out k=$k", @benchmark next!($s, $m))
        foreach(unsubscribe!, subs)
    end
    # combineLatest(PushNew) over k sources, one full round
    for k in (2, 5, 10, 20)
        sources = [Subject(Marginal) for _ in 1:k]
        combined = combineLatest(Tuple(sources), PushNew())
        sub = subscribe!(combined, Sink{Any}(0))
        vals = [mrg(NormalMeanVariance(randn(), 1.0)) for _ in 1:k]
        pushall(sources, vals) = (foreach(next!, sources, vals); nothing)
        record("rocket/combineLatest PushNew k=$k round", @benchmark $pushall($sources, $vals))
        unsubscribe!(sub)
    end
    # collectLatest over N sources, one full round (the free energy and array posteriors use it)
    for N in (100, 1000, 10000)
        sources = [Subject(Float64) for _ in 1:N]
        collected = collectLatest(Float64, Float64, sources, sum)
        sub = subscribe!(collected, Sink{Float64}(0))
        pushall(sources) = (foreach(s -> next!(s, 1.0), sources); nothing)
        pushall(sources)
        b = @benchmark $pushall($sources) samples = 20 evals = 1
        record("rocket/collectLatest N=$N round", b)
        unsubscribe!(sub)
    end
end

# The callback event the mapping builds on every call, even with no callbacks: its `result` is
# inferred `Any`, so the event's type is instantiated at run time.
if want("event")
    mapping = MessageMapping(NormalMeanVariance, Target{:out}(), Val((:μ, :v)), nothing, DefaultAlgorithm(), nothing, nothing, nothing)
    ms = (msg(NormalMeanVariance(1.0, 2.0)), msg(PointMass(0.5)))
    rr = Ref{Any}(NormalMeanVariance(1.0, 3.0)); ann = ReactiveMP.AnnotationDict()
    ev(mapping, ms, rr, ann) = ReactiveMP.AfterMessageRuleCallEvent(mapping, ms, nothing, rr[], ann, nothing, nothing)
    record("event/AfterMessageRuleCallEvent, result::Any", @benchmark $ev($mapping, $ms, $rr, $ann))
    record("event/generate_span_id((;))", @benchmark ReactiveMP.generate_span_id($((;))))
    rany = Ref{Any}(NormalMeanVariance(1.0, 3.0))
    record("ctor/Message from Any", @benchmark Message($rany[], false, false, $ann, nothing))
    record("ctor/AnnotationDict()", @benchmark ReactiveMP.AnnotationDict())
end

# engine graph loops, as scripts/benchmark_message_representation.jl built them
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
function rungraph(graph, n, iterations, data; fe = true)
    build = @timed begin
        randoms, y, nodes, watched = graph(n)
        foreach(v -> activate!(v, RandomVariableActivationOptions()), randoms)
        foreach(v -> activate!(v, DataVariableActivationOptions()), y)
        foreach(nd -> activate!(nd, FactorNodeActivationOptions()), nodes)
        subs = [subscribe!(get_stream_of_marginals(w), (_) -> nothing) for w in watched]
        fes = fe ? subscribe!(bethe_free_energy(Float64, nodes, [randoms..., y...]), (_) -> nothing) : nothing
    end
    stats = @timed for _ in 1:iterations
        foreach(new_observation!, y, data)
    end
    foreach(unsubscribe!, subs); isnothing(fes) || unsubscribe!(fes)
    return build, stats
end
if want("graph")
    for (name, graph, n, it, fe) in (
            ("graph/iid n=100", iid_graph, 100, 10, true),
            ("graph/iid n=1000", iid_graph, 1000, 10, true),
            ("graph/iid n=10000", iid_graph, 10000, 10, true),
            ("graph/iid n=1000 no-FE", iid_graph, 1000, 10, false),
            ("graph/chain n=300", chain_graph, 300, 10, true),
            ("graph/chain n=1000", chain_graph, 1000, 10, true),
            ("graph/chain n=1000 no-FE", chain_graph, 1000, 10, false),
        )
        data = randn(n)
        rungraph(graph, n, it, data; fe = fe)
        samples = [(GC.gc(); rungraph(graph, n, it, data; fe = fe)) for _ in 1:7]
        ts = map(s -> s[2].time * 1.0e9 / it, samples)
        record_raw(name * " per-iteration", minimum(ts), median(ts), -1, minimum(s -> s[2].bytes, samples) ÷ it)
        bs = map(s -> s[1].time * 1.0e9, samples)
        record_raw(name * " build+activate", minimum(bs), median(bs), -1, minimum(s -> s[1].bytes, samples))
    end
end

