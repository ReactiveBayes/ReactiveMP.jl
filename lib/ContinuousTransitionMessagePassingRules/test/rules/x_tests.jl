# From v6's `test/rules/continuous_transition/x_tests.jl`.

@testitem "rules:ContinuousTransition:x" tags = [:rules] begin
    using ContinuousTransitionMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, LinearAlgebra, Random
    using BayesBase: tiny

    diageye(n) = Matrix{Float64}(I, n, n)

    rng = MersenneTwister(42)

    @testset "Linear transformation" begin
        # the following rule is used for testing purposes only
        # It is derived separately by Thijs van de Laar
        function benchmark_rule_structured(q_y, q_W, mA, ΣA, UA)
            my, Vy = mean_cov(q_y)

            mW = mean(q_W)

            Λ = tr(mW * ΣA) * UA + mA' * inv(Vy + inv(mW)) * mA
            ξ = mA' * inv(Vy + inv(mW)) * my
            return MvNormalWeightedMeanPrecision(ξ, Λ)
        end

        @testset "Structured: (m_y::MultivariateNormalDistributionsFamily, q_a::MultivariateNormalDistributionsFamily, q_W::Any, meta::CTMeta)" begin
            for (dy, dx) in [(1, 3), (2, 3), (3, 2), (2, 2)]
                dydx = dy * dx
                transformation = (a) -> reshape(a, dy, dx)

                mA, ΣA, UA = rand(rng, dy, dx), diageye(dy), diageye(dx)

                Ly = rand(rng, dy, dy)
                μy, Σy = rand(rng, dy), Ly * Ly' + dydx * I

                qy = MvNormalMeanCovariance(μy, Σy)
                qa = MvNormalMeanCovariance(vec(mA), diageye(dydx))
                qW = Wishart(dy + 1, diageye(dy))

                @test_message_update_rule(
                    node = ContinuousTransition, target = :x, algorithm = CTVMP(transformation), atol = 1.0e-4,
                    cases = [(m = (y = qy,), q = (a = qa, W = qW)) => benchmark_rule_structured(qy, qW, mA, ΣA, UA)],
                )
            end
        end
    end

    @testset "Nonlinear transformation" begin
        @testset "Structured: (m_y::MultivariateNormalDistributionsFamily, q_a::Any, q_W::Any, meta::CTMeta)" begin
            dy, dx = 2, 2
            transformation = (a) -> [cos(a[1]) -sin(a[1]); sin(a[1]) cos(a[1])]

            μy, Σy = zeros(dy), diageye(dy)

            qy = MvNormalMeanCovariance(μy, Σy)
            qa = MvNormalMeanCovariance(zeros(1), tiny * diageye(1))
            qW = Wishart(dy + 1, diageye(dy))

            @test_message_update_rule(
                node = ContinuousTransition, target = :x, algorithm = CTVMP(transformation),
                cases = [(m = (y = qy,), q = (a = qa, W = qW)) => MvGaussianWeightedMeanPrecision(zeros(dx), 3 / 4 * diageye(dx))],
            )
        end
    end

    # the following rule is used for testing purposes only
    # It is derived separately by Thijs van de Laar
    function benchmark_rule_meanfield(q_y, q_W, mA, ΣA, UA)
        mW = mean(q_W)

        Λ = mA'mW * mA + tr(mW * ΣA) * UA
        ξ = mA' * mW * mean(q_y)
        return MvNormalWeightedMeanPrecision(ξ, Λ)
    end

    @testset "Mean-field: (q_y::Any, q_a::Any, q_W::Any, meta::CTMeta)" begin
        for (dy, dx) in [(1, 3), (2, 3), (3, 2), (2, 2)]
            dydx = dy * dx
            transformation = (a) -> reshape(a, dy, dx)

            mA, ΣA, UA = rand(rng, dy, dx), diageye(dy), diageye(dx)

            Ly = rand(rng, dy, dy)
            μy, Σy = rand(rng, dy), Ly * Ly' + dydx * I

            qy = MvNormalMeanCovariance(μy, Σy)
            qa = MvNormalMeanCovariance(vec(mA), diageye(dydx))
            qW = Wishart(dy + 1, diageye(dy))

            @test_message_update_rule(
                node = ContinuousTransition, target = :x, algorithm = CTVMP(transformation), atol = 1.0e-4,
                cases = [(q = (y = qy, a = qa, W = qW),) => benchmark_rule_meanfield(qy, qW, mA, ΣA, UA)],
            )
        end
    end
end
