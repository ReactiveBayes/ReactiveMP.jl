using BenchmarkTools
using Random

# Define a parent BenchmarkGroup to contain our suite
const SUITE = BenchmarkGroup()

SUITE["models"] = BenchmarkGroup([ "models", "ssm", "graphppl" ])

# Simple Linear Gaussian State Space Model Benchmarks 
# ------------------------------------------------------------ #
include("models/lgssm1.jl")

SUITE["models"]["lgssm1"] = BenchmarkGroup([ "linear", "gaussian", "ssm", "univariate" ])

for lgssm1_size in [ 100, 500 ]
    # Model creation benchmark
    SUITE["models"]["lgssm1"]["creation_$lgssm1_size"]  = @benchmarkable LGSSM1Benchmark.lgssm($lgssm1_size)
    # Inference benchmark
    SUITE["models"]["lgssm1"]["inference_$lgssm1_size"] = @benchmarkable LGSSM1Benchmark.benchmark(input) setup=(input=LGSSM1Benchmark.generate_input(MersenneTwister(1234), $lgssm1_size))
end
# ------------------------------------------------------------ #

# Multivariate Linear Gaussian State Space Model Benchmarks 
# ------------------------------------------------------------ #
include("models/lgssm2.jl")

SUITE["models"]["lgssm2"] = BenchmarkGroup([ "linear", "gaussian", "ssm", "multivariate" ])

for lgssm2_size in [ 100, 500 ]
    # Model creation benchmark
    SUITE["models"]["lgssm2"]["creation_$lgssm2_size"]  = @benchmarkable LGSSM2Benchmark.lgssm($lgssm2_size)
    # Inference benchmark
    SUITE["models"]["lgssm2"]["inference_$lgssm2_size"] = @benchmarkable LGSSM2Benchmark.benchmark(input) setup=(input=LGSSM2Benchmark.generate_input(MersenneTwister(1234), $lgssm2_size))
end
# ------------------------------------------------------------ #