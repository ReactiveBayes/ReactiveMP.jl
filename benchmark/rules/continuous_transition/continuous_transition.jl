using BenchmarkTools
using ReactiveMP
using BayesBase
using ExponentialFamily
using Random
using LinearAlgebra
using Distributions
using StableRNGs

import ReactiveMP: CTMeta, Marginal, Message, @call_rule, @call_marginalrule

"""
Creates test data for ContinuousTransition benchmarks.
Returns distributions and meta needed to call the rules.
"""
function create_ct_benchmark_data(dx, dy)
    rng = StableRNGs(42)
    da = dx * dy  # For linear transformation a -> reshape(a, dy, dx)

    # Transformation function
    transformation = a -> reshape(a, dy, dx)
    meta = CTMeta(transformation)

    # Create covariance matrices
    Lx = rand(rng, dx, dx)
    Ly = rand(rng, dy, dy)
    La = rand(rng, da, da)

    μx, Σx = rand(rng, dx), Lx * Lx' + dx * I
    μy, Σy = rand(rng, dy), Ly * Ly' + dy * I
    μa, Σa = rand(rng, da), La * La' + da * I

    # Create distributions for mean-field factorization
    q_y = MvNormalMeanCovariance(μy, Σy)
    q_x = MvNormalMeanCovariance(μx, Σx)
    q_a = MvNormalMeanCovariance(μa, Σa)
    q_W = Wishart(dy + 1, Matrix{Float64}(I, dy, dy))

    # Create joint distribution for structured factorization
    q_y_x = MvNormalMeanCovariance([μy; μx], [Σy zeros(dy, dx); zeros(dx, dy) Σx])

    # Create messages for marginal rule
    m_y = MvNormalMeanCovariance(μy, Σy)
    m_x = MvNormalMeanCovariance(μx, Σx)

    return (meta = meta, q_y = q_y, q_x = q_x, q_a = q_a, q_W = q_W, q_y_x = q_y_x, m_y = m_y, m_x = m_x)
end

"""
Adds ContinuousTransition rule benchmarks to the suite.
"""
function add_continuous_transition_rule_benchmarks(SUITE)
    SUITE["ContinuousTransition"] = BenchmarkGroup()

    add_continuous_transition_a_benchmarks(SUITE["ContinuousTransition"])
    add_continuous_transition_marginals_benchmarks(SUITE["ContinuousTransition"])
end

function add_continuous_transition_a_benchmarks(SUITE)
    SUITE["a"] = BenchmarkGroup(["Rules", "ContinuousTransition"])

    # Test dimensions: (dx, dy)
    test_dims = [(5, 5), (10, 10), (20, 20), (30, 30)]

    for (dx, dy) in test_dims
        data = create_ct_benchmark_data(dx, dy)

        # Structured VMP: q(y,x) joint
        SUITE["a"]["Structured"]["dx=$(dx), dy=$(dy)"] = @benchmarkable begin
            @call_rule ContinuousTransition(:a, Marginalisation) (q_y_x = $data.q_y_x, q_a = $data.q_a, q_W = $data.q_W, meta = $data.meta)
        end

        # Mean-field VMP: q(y)q(x)q(a)q(W)
        SUITE["a"]["Mean-field"]["dx=$(dx), dy=$(dy)"] = @benchmarkable begin
            @call_rule ContinuousTransition(:a, Marginalisation) (q_y = $data.q_y, q_x = $data.q_x, q_a = $data.q_a, q_W = $data.q_W, meta = $data.meta)
        end
    end
end

function add_continuous_transition_marginals_benchmarks(SUITE)
    SUITE["marginals"] = BenchmarkGroup(["Rules", "ContinuousTransition"])

    # Test dimensions: (dx, dy)
    test_dims = [(5, 5), (10, 10), (20, 20), (30, 30)]

    for (dx, dy) in test_dims
        data = create_ct_benchmark_data(dx, dy)

        # y_x marginal rule
        SUITE["marginals"]["y_x"]["dx=$(dx), dy=$(dy)"] = @benchmarkable begin
            @call_marginalrule ContinuousTransition(:y_x) (m_y = $data.m_y, m_x = $data.m_x, q_a = $data.q_a, q_W = $data.q_W, meta = $data.meta)
        end
    end
end
