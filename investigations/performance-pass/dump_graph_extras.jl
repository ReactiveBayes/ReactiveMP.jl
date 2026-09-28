# Dumps, per factor node in creation order, its resolved factorisation clusters and its meta
# (algorithm) after the `graph` stage: julia --project=<env> dump_graph_extras.jl <variant> <outdir> <model> <n>
const NO_MAIN = true
const DV, DOUT, DM, DN = ARGS[1], ARGS[2], ARGS[3], parse(Int, ARGS[4])
empty!(ARGS); append!(ARGS, [DV, "dump", "/dev/null", DM])
include(joinpath(@__DIR__, "bench_creation.jl"))
using Serialization
import GraphPPL
gen, data, cons, meta, init = setup(DM, DN)
m = RxInfer.getmodel(create("graph", gen, data, cons, meta, init))
rows = []
GraphPPL.factor_nodes(m) do label, nd
    clusters = GraphPPL.hasextra(nd, GraphPPL.VariationalConstraintsFactorizationIndicesKey) ? GraphPPL.getextra(nd, GraphPPL.VariationalConstraintsFactorizationIndicesKey) : nothing
    mt = GraphPPL.hasextra(nd, GraphPPL.MetaExtraKey) ? repr(GraphPPL.getextra(nd, GraphPPL.MetaExtraKey)) : nothing
    push!(rows, (label.global_counter, repr(GraphPPL.fform(GraphPPL.getproperties(nd))), clusters === nothing ? nothing : map(collect, clusters), mt))
end
serialize(joinpath(DOUT, "$(DV)_$(DM)_$(DN).jls"), rows)
println(DV, " ", DM, " ", length(rows), " factor nodes; with meta: ", count(r -> r[4] !== nothing, rows))
