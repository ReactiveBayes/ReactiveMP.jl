using BenchmarkTools
using Random

# Define a parent BenchmarkGroup to contain our suite
const SUITE = BenchmarkGroup()

SUITE["models"] = BenchmarkGroup([ "models", "ssm", "graphppl" ])

rng = MersenneTwister(1234)

# Simple Linear Gaussian State Space Model Benchmarks 
# ------------------------------------------------------------ #
include("models/lgssm1.jl")

SUITE["models"]["lgssm1"] = BenchmarkGroup([ "linear", "gaussian", "ssm", "univariate" ])

for lgssm1_size in [ 100, 200, 300, 400, 500 ]
    # Model creation benchmark
    SUITE["models"]["lgssm1"]["creation_100"]  = @benchmarkable LGSSM1Benchmark.lgssm($lgssm1_size)
    # Inference benchmark
    SUITE["models"]["lgssm1"]["inference_100"] = @benchmarkable LGSSM1Benchmark.benchmark(input) setup=(input=LGSSM1Benchmark.generate_input(rng, lgssm1_size))
end
# ------------------------------------------------------------ #

# Multivariate Linear Gaussian State Space Model Benchmarks 
# ------------------------------------------------------------ #
include("models/lgssm2.jl")

SUITE["models"]["lgssm2"] = BenchmarkGroup([ "linear", "gaussian", "ssm", "multivariate" ])

for lgssm2_size in [ 100, 200, 300, 400, 500 ]
    # Model creation benchmark
    SUITE["models"]["lgssm2"]["creation_100"]  = @benchmarkable LGSSM2Benchmark.lgssm($lgssm2_size)
    # Inference benchmark
    SUITE["models"]["lgssm2"]["inference_100"] = @benchmarkable LGSSM2Benchmark.benchmark(input) setup=(input=LGSSM1Benchmark.generate_input(rng, lgssm2_size))
end
# ------------------------------------------------------------ #