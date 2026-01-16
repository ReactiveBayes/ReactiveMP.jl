#!/usr/bin/env julia
#=
ContinuousTransition Rules Benchmark Script

Run this script to benchmark the ContinuousTransition rules performance.
Can be executed on different branches to compare optimizations.

Usage:
    julia --project=. benchmark/continuous_transition_bench.jl
    julia --project=. benchmark/continuous_transition_bench.jl quick    # Quick mode (only small dims)

Output: Performance table showing timings for each rule and dimension.
=#

using Pkg
Pkg.activate(dirname(@__DIR__))

using BenchmarkTools
using ReactiveMP
using BayesBase
using ExponentialFamily
using Random
using LinearAlgebra
using Distributions
using Printf

import ReactiveMP: CTMeta, @call_rule, @call_marginalrule

# ============================================================================
#  Test Data Generation
# ============================================================================

function create_benchmark_data(dx, dy)
    rng = MersenneTwister(42)
    da = dx * dy

    transformation = a -> reshape(a, dy, dx)
    meta = CTMeta(transformation)

    Lx = rand(rng, dx, dx)
    Ly = rand(rng, dy, dy)
    La = rand(rng, da, da)

    μx, Σx = rand(rng, dx), Lx * Lx' + dx * I
    μy, Σy = rand(rng, dy), Ly * Ly' + dy * I
    μa, Σa = rand(rng, da), La * La' + da * I

    q_y = MvNormalMeanCovariance(μy, Σy)
    q_x = MvNormalMeanCovariance(μx, Σx)
    q_a = MvNormalMeanCovariance(μa, Σa)
    q_W = Wishart(dy + 1, Matrix{Float64}(I, dy, dy))
    q_y_x = MvNormalMeanCovariance([μy; μx], [Σy zeros(dy, dx); zeros(dx, dy) Σx])

    m_y = MvNormalMeanCovariance(μy, Σy)
    m_x = MvNormalMeanCovariance(μx, Σx)

    return (meta = meta, q_y = q_y, q_x = q_x, q_a = q_a, q_W = q_W, q_y_x = q_y_x, m_y = m_y, m_x = m_x)
end

# ============================================================================
#  Benchmark Functions
# ============================================================================

function bench_a_structured(data)
    @call_rule ContinuousTransition(:a, Marginalisation) (q_y_x = data.q_y_x, q_a = data.q_a, q_W = data.q_W, meta = data.meta)
end

function bench_a_meanfield(data)
    @call_rule ContinuousTransition(:a, Marginalisation) (q_y = data.q_y, q_x = data.q_x, q_a = data.q_a, q_W = data.q_W, meta = data.meta)
end

function bench_marginal_y_x(data)
    @call_marginalrule ContinuousTransition(:y_x) (m_y = data.m_y, m_x = data.m_x, q_a = data.q_a, q_W = data.q_W, meta = data.meta)
end

# ============================================================================
#  Benchmark Runner
# ============================================================================

function run_benchmarks(; quick_mode = false)
    println()
    println("=" ^ 80)
    println("  ContinuousTransition Rules Benchmark")
    println("  Branch: ", strip(read(`git rev-parse --abbrev-ref HEAD`, String)))
    println("  Commit: ", strip(read(`git rev-parse --short HEAD`, String)))
    println("=" ^ 80)
    println()

    if quick_mode
        test_dims = [(10, 10), (20, 20)]
        println("  Mode: QUICK (limited dimensions)")
    else
        test_dims = [(5, 5), (10, 10), (20, 20), (30, 30), (40, 40)]
        println("  Mode: FULL")
    end
    println()

    # Results storage
    results = Dict{String, Vector{Tuple{Int, Int, Float64}}}("a_structured" => [], "a_meanfield" => [], "marginal_y_x" => [])

    for (dx, dy) in test_dims
        println("-" ^ 60)
        @printf("  Benchmarking: dx=%d, dy=%d (da=%d)\n", dx, dy, dx*dy)
        println("-" ^ 60)

        data = create_benchmark_data(dx, dy)

        # Warm-up calls
        bench_a_structured(data)
        bench_a_meanfield(data)
        bench_marginal_y_x(data)

        # Benchmark a.jl structured
        t = @belapsed bench_a_structured($data)
        push!(results["a_structured"], (dx, dy, t * 1e6))
        @printf("    a.jl Structured:  %10.2f μs\n", t * 1e6)

        # Benchmark a.jl mean-field
        t = @belapsed bench_a_meanfield($data)
        push!(results["a_meanfield"], (dx, dy, t * 1e6))
        @printf("    a.jl Mean-field:  %10.2f μs\n", t * 1e6)

        # Benchmark marginals.jl
        t = @belapsed bench_marginal_y_x($data)
        push!(results["marginal_y_x"], (dx, dy, t * 1e6))
        @printf("    marginals.jl y_x: %10.2f μs\n", t * 1e6)

        println()
    end

    # Print summary table
    println("=" ^ 80)
    println("  SUMMARY TABLE (times in μs)")
    println("=" ^ 80)
    println()

    # Header
    @printf("  %-12s", "Dimensions")
    for rule in ["a_structured", "a_meanfield", "marginal_y_x"]
        @printf(" | %14s", replace(rule, "_" => " "))
    end
    println()
    println("  " * "-" ^ 12, " | ", "-" ^ 14, " | ", "-" ^ 14, " | ", "-" ^ 14)

    # Data rows
    for i in eachindex(test_dims)
        dx, dy = test_dims[i]
        @printf("  %4d × %-4d ", dx, dy)
        @printf(" | %14.2f", results["a_structured"][i][3])
        @printf(" | %14.2f", results["a_meanfield"][i][3])
        @printf(" | %14.2f", results["marginal_y_x"][i][3])
        println()
    end

    println()
    println("=" ^ 80)
    println("  Benchmark Complete")
    println("=" ^ 80)
    println()

    return results
end

# ============================================================================
#  Main
# ============================================================================

quick_mode = length(ARGS) > 0 && ARGS[1] == "quick"
run_benchmarks(quick_mode = quick_mode)
