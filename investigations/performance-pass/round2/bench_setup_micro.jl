# Node creation and activation per node, without RxInfer, on a v7 variant.
#   julia --project=<variant>/env bench_setup_micro.jl <variant> <tag> <outfile>
# One TSV line per case: variant, tag, case, min ns per node, median ns per node, allocations per
# node, bytes per node. Each sample builds N nodes of one shape on fresh variables.
using ReactiveMP, Rocket, BayesBase, ExponentialFamily, Distributions, StandardMessagePassingRules, Statistics
import ReactiveMP: factornode, activate!, FactorNodeActivationOptions, randomvar, datavar, constvar, israndom, isdata,
    RandomVariableActivationOptions, DataVariableActivationOptions, MessageProductContext
import MessagePassingRulesBase

const VARIANT = get(ARGS, 1, "?")
const TAG = get(ARGS, 2, "?")
const OUT = get(ARGS, 3, "/dev/stdout")
const N = 1000

function record(name, samples)
    ts = map(s -> s.time, samples); a = minimum(map(s -> s.allocs, samples)); b = minimum(map(s -> s.bytes, samples))
    line = join((VARIANT, TAG, name, round(1.0e9 * minimum(ts) / N; digits = 1), round(1.0e9 * median(ts) / N; digits = 1), round(a / N; digits = 1), round(b / N; digits = 1)), '\t')
    println(line)
    OUT == "/dev/stdout" || open(io -> println(io, line), OUT, "a")
    return nothing
end

function timed_allocs(f)
    GC.gc()
    s = @timed f()
    c = Base.gc_alloc_count(s.gcstats)
    return (time = s.time - s.gctime, allocs = c, bytes = s.bytes)
end

# (fform, interface keys, factorisation) of the shapes the benchmarks' models create
const SHAPES = [
    ("NMV (out, μ, v) BP", NormalMeanVariance, (:out, :μ, :v), ((:out, :μ), (:v,)), (:r, :r, :c)),
    ("NMP (out, μ, τ) MF", NormalMeanPrecision, (:out, :μ, :τ), ((:out,), (:μ,), (:τ,)), (:d, :r, :r)),
    ("Bernoulli (out, p)", Bernoulli, (:out, :p), ((:out, :p),), (:d, :r)),
    ("NormalMixture (out, switch, m1..2, p1..2) MF", NormalMixture, (:out, :switch, (:m, 1), (:m, 2), (:p, 1), (:p, 2)), ((:out,), (:switch,), ((:m, 1),), ((:m, 2),), ((:p, 1),), ((:p, 2),)), (:d, :r, :r, :r, :r, :r)),
]

mkvar(kind) = kind === :r ? randomvar() : kind === :d ? datavar() : constvar(1.0)

function shape_inputs(keys, kinds)
    return [[(k, mkvar(kind)) for (k, kind) in zip(keys, kinds)] for _ in 1:N]
end

isdefined(Main, :NO_RUN) || for (name, fform, keys, factorisation, kinds) in SHAPES
    create = inputs -> [factornode(fform, i, factorisation) for i in inputs]
    create(shape_inputs(keys, kinds))
    record("create/" * name, [(inputs = shape_inputs(keys, kinds); timed_allocs(() -> create(inputs))) for _ in 1:15])
    options = FactorNodeActivationOptions()
    activate_all = nodes -> foreach(node -> activate!(node, options), nodes)
    # the variables are activated first, as RxInfer does, and not timed
    function prepared_nodes()
        inputs = shape_inputs(keys, kinds)
        nodes = create(inputs)
        product = MessageProductContext()
        for i in inputs, (_, v) in i
            israndom(v) && activate!(v, RandomVariableActivationOptions(nothing, product, product))
            isdata(v) && activate!(v, DataVariableActivationOptions(true, false, nothing, nothing))
        end
        return nodes
    end
    activate_all(prepared_nodes())
    record("activate/" * name, [(nodes = prepared_nodes(); timed_allocs(() -> activate_all(nodes))) for _ in 1:15])
end
