using BenchmarkTools
using Random

# Define a parent BenchmarkGroup to contain our suite
const SUITE = BenchmarkGroup()

SUITE["models"] = BenchmarkGroup([ "models", "ssm", "graphppl" ])

include("models/lgssm1.jl")

SUITE["models"]["lgssm1"] = BenchmarkGroup([ "linear", "gaussian", "ssm" ])

rng = MersenneTwister(1234)

SUITE["models"]["lgssm1"]["n_50"]  = @benchmarkable LGSSM1Benchmark.benchmark(input) setup=(input=LGSSM1Benchmark.generate_input(rng, 50))
SUITE["models"]["lgssm1"]["n_100"] = @benchmarkable LGSSM1Benchmark.benchmark(input) setup=(input=LGSSM1Benchmark.generate_input(rng, 100))
SUITE["models"]["lgssm1"]["n_200"] = @benchmarkable LGSSM1Benchmark.benchmark(input) setup=(input=LGSSM1Benchmark.generate_input(rng, 200))
SUITE["models"]["lgssm1"]["n_500"] = @benchmarkable LGSSM1Benchmark.benchmark(input) setup=(input=LGSSM1Benchmark.generate_input(rng, 500))