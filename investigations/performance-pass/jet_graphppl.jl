# JET optimisation analysis of model creation, GraphPPL frames only.
#   julia --project=<env> jet_graphppl.jl <model> <stage> <outfile>
const NO_MAIN = true
const JM, JSTAGE, JOUT = ARGS[1], ARGS[2], ARGS[3]
empty!(ARGS); append!(ARGS, ["jet", "jet", "/dev/null", JM])
include(joinpath(@__DIR__, "bench_creation.jl"))
using JET
import GraphPPL
gen, data, cons, meta, init = setup(JM, 100)
create(JSTAGE, gen, data, cons, meta, init)
report = JET.report_opt(
    create, (typeof(JSTAGE), typeof(gen), typeof(data), typeof(cons), typeof(meta), typeof(init));
    target_modules = (GraphPPL,)
)
reports = JET.get_reports(report)
open(JOUT, "w") do io
    println(io, "reports: ", length(reports))
    show(IOContext(io, :limit => false, :displaysize => (10000, 250)), report)
end
println("reports: ", length(reports))
