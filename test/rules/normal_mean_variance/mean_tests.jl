
@testitem "rules:NormalMeanVariance:mean" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules

    @testset "Belief Propagation: (m_out::PointMass, m_v::PointMass)" begin
        @test_rules [check_type_promotion = true] NormalMeanVariance(
            :μ, Marginalisation
        ) [
            (
                input = (m_out = PointMass(-1.0), m_v = PointMass(2.0)),
                output = NormalMeanVariance(-1.0, 2.0),
            ),
            (
                input = (m_out = PointMass(1.0), m_v = PointMass(2.0)),
                output = NormalMeanVariance(1.0, 2.0),
            ),
            (
                input = (m_out = PointMass(2.0), m_v = PointMass(1.0)),
                output = NormalMeanVariance(2.0, 1.0),
            ),
        ]
    end

    @testset "Belief Propagation: (m_out::UnivariateNormalDistributionsFamily, m_v::PointMass)" begin
        @test_rules [check_type_promotion = true] NormalMeanVariance(
            :μ, Marginalisation
        ) [
            (
                input = (
                    m_out = NormalMeanVariance(0.0, 1.0), m_v = PointMass(2.0)
                ),
                output = NormalMeanVariance(0.0, 3.0),
            ),
            (
                input = (
                    m_out = NormalMeanVariance(-1.0, 1.0), m_v = PointMass(1.5)
                ),
                output = NormalMeanVariance(-1.0, 2.5),
            ),
            (
                input = (
                    m_out = NormalMeanVariance(2.0, 0.5), m_v = PointMass(1.0)
                ),
                output = NormalMeanVariance(2.0, 1.5),
            ),
        ]

        @test_rules [check_type_promotion = true] NormalMeanVariance(
            :μ, Marginalisation
        ) [
            (
                input = (
                    m_out = NormalMeanPrecision(0.0, 1.0), m_v = PointMass(2.0)
                ),
                output = NormalMeanVariance(0.0, 3.0),
            ),
            (
                input = (
                    m_out = NormalMeanPrecision(-1.0, 1.0), m_v = PointMass(1.5)
                ),
                output = NormalMeanVariance(-1.0, 2.5),
            ),
            (
                input = (
                    m_out = NormalMeanPrecision(2.0, 0.5), m_v = PointMass(1.0)
                ),
                output = NormalMeanVariance(2.0, 3.0),
            ),
        ]

        @test_rules [check_type_promotion = true] NormalMeanVariance(
            :μ, Marginalisation
        ) [
            (
                input = (
                    m_out = NormalWeightedMeanPrecision(0.0, 1.0),
                    m_v = PointMass(2.0),
                ),
                output = NormalMeanVariance(0.0, 3.0),
            ),
            (
                input = (
                    m_out = NormalWeightedMeanPrecision(-1.0, 1.0),
                    m_v = PointMass(1.5),
                ),
                output = NormalMeanVariance(-1.0, 2.5),
            ),
            (
                input = (
                    m_out = NormalWeightedMeanPrecision(2.0, 0.5),
                    m_v = PointMass(1.0),
                ),
                output = NormalMeanVariance(4.0, 3.0),
            ),
        ]
    end

    @testset "Belief Propagation: (m_out::UnivariateNormalDistributionsFamily, q_v::PointMass)" begin
        @test_rules [check_type_promotion = true] NormalMeanVariance(
            :μ, Marginalisation
        ) [
            (
                input = (
                    m_out = NormalMeanVariance(0.0, 1.0), q_v = PointMass(2.0)
                ),
                output = NormalMeanVariance(0.0, 3.0),
            ),
            (
                input = (
                    m_out = NormalMeanVariance(-1.0, 1.0), q_v = PointMass(1.5)
                ),
                output = NormalMeanVariance(-1.0, 2.5),
            ),
            (
                input = (
                    m_out = NormalMeanVariance(2.0, 0.5), q_v = PointMass(1.0)
                ),
                output = NormalMeanVariance(2.0, 1.5),
            ),
        ]

        @test_rules [check_type_promotion = true] NormalMeanVariance(
            :μ, Marginalisation
        ) [
            (
                input = (
                    m_out = NormalMeanPrecision(0.0, 1.0), q_v = PointMass(2.0)
                ),
                output = NormalMeanVariance(0.0, 3.0),
            ),
            (
                input = (
                    m_out = NormalMeanPrecision(-1.0, 1.0), q_v = PointMass(1.5)
                ),
                output = NormalMeanVariance(-1.0, 2.5),
            ),
            (
                input = (
                    m_out = NormalMeanPrecision(2.0, 0.5), q_v = PointMass(1.0)
                ),
                output = NormalMeanVariance(2.0, 3.0),
            ),
        ]

        @test_rules [check_type_promotion = true] NormalMeanVariance(
            :μ, Marginalisation
        ) [
            (
                input = (
                    m_out = NormalWeightedMeanPrecision(0.0, 1.0),
                    q_v = PointMass(2.0),
                ),
                output = NormalMeanVariance(0.0, 3.0),
            ),
            (
                input = (
                    m_out = NormalWeightedMeanPrecision(-1.0, 1.0),
                    q_v = PointMass(1.5),
                ),
                output = NormalMeanVariance(-1.0, 2.5),
            ),
            (
                input = (
                    m_out = NormalWeightedMeanPrecision(2.0, 0.5),
                    q_v = PointMass(1.0),
                ),
                output = NormalMeanVariance(4.0, 3.0),
            ),
        ]
    end

    @testset "Variational: (m_out::PointMass, q_v::Any)" begin
        @test_rules [check_type_promotion = true] NormalMeanVariance(
            :μ, Marginalisation
        ) [
            (
                input = (
                    m_out = PointMass(-1.0), q_v = GammaShapeRate(1.0, 1.0)
                ),
                output = NormalMeanVariance(-1.0, 1.0),
            ),
            (
                input = (
                    m_out = PointMass(1.0), q_v = GammaShapeScale(1.0, 1.0)
                ),
                output = NormalMeanVariance(1.0, 1.0),
            ),
            (
                input = (m_out = PointMass(2.0), q_v = PointMass(1.0)),
                output = NormalMeanVariance(2.0, 1.0),
            ),
        ]
    end

    @testset "Variational: (q_out::Any, q_v::Any)" begin
        @test_rules [check_type_promotion = true] NormalMeanVariance(
            :μ, Marginalisation
        ) [
            (
                input = (q_out = PointMass(-1.0), q_v = PointMass(2.0)),
                output = NormalMeanVariance(-1.0, 2.0),
            ),
            (
                input = (q_out = PointMass(1.0), q_v = PointMass(2.0)),
                output = NormalMeanVariance(1.0, 2.0),
            ),
            (
                input = (q_out = PointMass(2.0), q_v = PointMass(1.0)),
                output = NormalMeanVariance(2.0, 1.0),
            ),
        ]

        @test_rules [check_type_promotion = true] NormalMeanVariance(
            :μ, Marginalisation
        ) [
            (
                input = (
                    q_out = NormalMeanVariance(-1.0, 2.0), q_v = PointMass(2.0)
                ),
                output = NormalMeanVariance(-1.0, 2.0),
            ),
            (
                input = (
                    q_out = NormalMeanPrecision(1.0, 4.0), q_v = PointMass(3.0)
                ),
                output = NormalMeanVariance(1.0, 3.0),
            ),
            (
                input = (
                    q_out = NormalWeightedMeanPrecision(2.0, 4.0),
                    q_v = PointMass(1.0),
                ),
                output = NormalMeanVariance(0.5, 1.0),
            ),
        ]

        @test_rules [check_type_promotion = true] NormalMeanVariance(
            :μ, Marginalisation
        ) [
            (
                input = (q_out = PointMass(-1.0), q_v = InverseGamma(2.0, 1.0)),
                output = NormalMeanVariance(-1.0, 1.0),
            ),
            (
                input = (q_out = PointMass(1.0), q_v = InverseGamma(4.0, 2.0)),
                output = NormalMeanVariance(1.0, 2.0 / 3.0),
            ),
            (
                input = (q_out = PointMass(2.0), q_v = InverseGamma(4.0, 6.0)),
                output = NormalMeanVariance(2.0, 2.0),
            ),
        ]
    end
end

@testitem "rules:NormalMeanVariance:mean:generic VMP" begin
    using ReactiveMP, BayesBase, ExponentialFamily, Distributions, StableRNGs

    import ReactiveMP: @call_rule

    include("../../generic_vmp_rule.jl")

    rng = StableRNG(42)
    q_out = NormalMeanVariance(1.5, 0.7)
    q_v = InverseGamma(10.0, 4.0)

    # The factor arguments are (out, μ, v); leave the mean interface free.
    log_node_function(out, μ, v) = logpdf(NormalMeanVariance(μ, v), out)
    log_samples = test_generic_vmp_rule(
        rng, log_node_function, 2, 100_000, q_out, nothing, q_v
    )
    message = @call_rule NormalMeanVariance(:μ, Marginalisation) (
        q_out = q_out, q_v = q_v
    )

    # Subtract a reference value to remove the unknown additive log constant.
    μ_reference = mean(q_out)
    sampled_reference = log_samples(μ_reference)
    rule_reference = logpdf(message, μ_reference)
    μ_values = range(-0.5, 3.5; length = 9)

    # Normal approximation with a false-failure target of 1e-6 across the grid.
    # Bonferroni adjustment does not require independent grid comparisons.
    α = 1e-6
    z = quantile(Normal(), 1 - α / (2 * length(μ_values)))
    for μ in μ_values
        # Pair samples before estimating uncertainty: both evaluations use the
        # same draws, so their Monte Carlo errors are correlated.
        differences = log_samples(μ) .- sampled_reference
        sampled_difference = mean(differences)
        standard_error = std(differences) / sqrt(length(differences))
        rule_difference = logpdf(message, μ) - rule_reference
        @test abs(sampled_difference - rule_difference) <= z * standard_error
    end
end
