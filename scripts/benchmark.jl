using PkgBenchmark, BenchmarkCI, Base64

BenchmarkCI.judge(; target = PkgBenchmark.BenchmarkConfig(; id = nothing, juliacmd = `julia -O3 --startup-file=no`))
BenchmarkCI.displayjudgement()

if get(ENV, "PUBLISH_BENCHMARKS", "FALSE") == "TRUE" && haskey(ENV, "BENCHMARK_KEY")
    BenchmarkCI.pushresult(;
        url = "git@github.com:biaslab/ReactiveMP.jl.git",
        title = "Benchmark result (via Travis)",
        branch = "gh-benchmarks",
        sshkey = String(Base64.base64decode(ENV["BENCHMARK_KEY"]))
    )
end
