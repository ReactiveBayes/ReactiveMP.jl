using BenchmarkTools

# Define a parent BenchmarkGroup to contain our suite
const suite = BenchmarkGroup()

suite["models"] = BenchmarkGroup([ "models", "ssm", "graphppl" ])

include("models/lgssm1.jl")

suite["models"]["lgssm1"] = BenchmarkGroup([ "linear", "gaussian", "ssm" ])

rng = MersenneTwister(1234)

suite["models"]["lgssm1"]["n_50"]  = @benchmarkable LGSSM1Benchmark.benchmark(input) setup=(input=LGSSM1Benchmark.generate_input(rng, 50))
suite["models"]["lgssm1"]["n_100"] = @benchmarkable LGSSM1Benchmark.benchmark(input) setup=(input=LGSSM1Benchmark.generate_input(rng, 100))
suite["models"]["lgssm1"]["n_200"] = @benchmarkable LGSSM1Benchmark.benchmark(input) setup=(input=LGSSM1Benchmark.generate_input(rng, 200))
suite["models"]["lgssm1"]["n_500"] = @benchmarkable LGSSM1Benchmark.benchmark(input) setup=(input=LGSSM1Benchmark.generate_input(rng, 500))