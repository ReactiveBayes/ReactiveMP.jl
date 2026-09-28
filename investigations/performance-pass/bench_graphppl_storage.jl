# P8b sizing: the cost of GraphPPL's storage on a built model (ssm1, n = 10000, full plugins).
#   julia --project=<env> bench_graphppl_storage.jl <variant> <outfile>
const NO_MAIN = true
const SV, SOUT = ARGS[1], ARGS[2]
empty!(ARGS); append!(ARGS, [SV, "storage", "/dev/null", "ssm1"])
include(joinpath(@__DIR__, "bench_creation.jl"))
using BenchmarkTools
import GraphPPL
gen, data, cons, meta, init = setup("ssm1", 10_000)
model = create("full", gen, data, cons, meta, init)
fgraph = RxInfer.getmodel(model)
labs = collect(GraphPPL.labels(fgraph))
nodes = [fgraph[l] for l in labs]                       # what a Vector{NodeData} store would hold
key = RxInfer.ReactiveMPExtraFactorNodeKey
fnodes = filter(GraphPPL.is_factor, nodes)
emit2(name, b) = (line = join((SV, name, BenchmarkTools.minimum(b).time / length(labs), BenchmarkTools.minimum(b).allocs), '\t'); println(line); open(io -> println(io, line), SOUT, "a"))
sumlookup(m, ls) = (
    c = 0; for l in ls
        c += GraphPPL.is_factor(m[l])
    end; c
)
sumvector(ns) = (
    c = 0; for n in ns
        c += GraphPPL.is_factor(n)
    end; c
)
iterlabels(m) = (
    c = 0; GraphPPL.factor_nodes(m) do l, d
        c += 1
    end; c
)
getextras(ns, k) = (
    c = 0; for n in ns
        c += GraphPPL.hasextra(n, k) ? 1 : 0; GraphPPL.getextra(n, k)
    end; c
)
emit2("model[label] per node (hashed)", @benchmark sumlookup($fgraph, $labs))
emit2("Vector{NodeData} per node", @benchmark sumvector($nodes))
emit2("factor_nodes(callback) per node", @benchmark iterlabels($fgraph))
emit2("hasextra+getextra(typed key) per factor node", @benchmark getextras($fnodes, $key))
