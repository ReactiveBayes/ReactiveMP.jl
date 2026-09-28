# Time to first model and GraphPPL specialisations for a model of K distinct `~` statements.
#   julia --project=<env> bench_ttfx_creation.jl <variant> <tag> <outfile> [K]
const T0 = time()
using RxInfer
import GraphPPL
const TLOAD = time() - T0
const VARIANT, TAG, OUT = ARGS[1], ARGS[2], ARGS[3]
const K = length(ARGS) > 3 ? parse(Int, ARGS[4]) : 40

# a1 ~ Normal(mean = 0.0, variance = 1.0); a_k ~ Normal(mean = a_{k-1}, variance = k); y ~ Normal(mean = a_K, variance = 1.0)
body = Expr(:block, :(a1 ~ Normal(mean = 0.0, variance = 1.0)))
for k in 2:K
    push!(body.args, :($(Symbol(:a, k)) ~ Normal(mean = $(Symbol(:a, k - 1)), variance = $(Float64(k)))))
end
push!(body.args, :(y ~ Normal(mean = $(Symbol(:a, K)), variance = 1.0)))
@eval @model function wide(y)
    $body
end

function create(stage)
    plugins = stage == "graph" ? GraphPPL.PluginsCollection(GraphPPL.VariationalConstraintsPlugin(nothing), GraphPPL.MetaPlugin(nothing)) :
        GraphPPL.PluginsCollection(
            GraphPPL.VariationalConstraintsPlugin(nothing), GraphPPL.MetaPlugin(nothing), RxInfer.InitializationPlugin(nothing),
            RxInfer.ReactiveMPInferencePlugin(RxInfer.setwarn(convert(RxInfer.ReactiveMPInferenceOptions, nothing), true))
        )
    model = GraphPPL.with_backend(GraphPPL.with_plugins(wide(), plugins), RxInfer.ReactiveMPGraphPPLBackend(RxInfer.Static.static(false)))
    return RxInfer.create_model(model | (y = 1.0,))
end

function nspecs(mod)
    n = 0
    for name in names(mod; all = true)
        isdefined(mod, name) || continue
        f = getfield(mod, name)
        f isa Function || continue
        for m in methods(f)
            m.module === mod || continue
            n += count(_ -> true, Base.specializations(m))
        end
    end
    return n
end

s0 = nspecs(GraphPPL)
t_graph = @elapsed create("graph")
s1 = nspecs(GraphPPL)
t_full = @elapsed create("full")
s2 = nspecs(GraphPPL)
t_again = @elapsed create("full")
line = join((VARIANT, TAG, K, TLOAD, t_graph, t_full, t_again, s1 - s0, s2 - s0), '\t')
println(line)
open(io -> println(io, line), OUT, "a")
