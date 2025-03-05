using ReactiveMP
using BenchmarkTools
using PkgBenchmark
using Dates

mkpath("./benchmark_logs")

if isempty(ARGS)
    result = PkgBenchmark.benchmarkpkg(ReactiveMP)
    export_markdown("./benchmark_logs/benchmark_$(now()).md", result)
    export_markdown("./benchmark_logs/last.md", result)
else
    name = first(ARGS)
    result = BenchmarkTools.judge(ReactiveMP, name; judgekwargs = Dict(:time_tolerance => 0.1, :memory_tolerance => 0.05))
    export_markdown("./benchmark_logs/benchmark_vs_$(name)_result.md", result)
end