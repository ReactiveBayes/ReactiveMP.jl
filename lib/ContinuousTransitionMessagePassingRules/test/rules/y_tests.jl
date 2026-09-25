@testitem "rules:ContinuousTransition:y" tags = [:rules] begin
    using ContinuousTransitionMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, LinearAlgebra, Random
    using BayesBase: tiny

    diageye(n) = Matrix{Float64}(I, n, n)

    rng = MersenneTwister(42)

    @testset "Linear transformation" begin
        # the following rule is used for testing purposes only
        # It is derived separately by Thijs van de Laar
        function benchmark_rule(q_x, q_W, mA)
            mx, Vx = mean_cov(q_x)
            mW = mean(q_W)
            return MvNormalMeanCovariance(mA * mx, mA * Vx * mA' + inv(mW))
        end

        @testset "Structured: (m_x::MultivariateNormalDistributionsFamily, q_a::MultivariateNormalDistributionsFamily, q_W::Any, algo::CTVMP)" begin
            for (dy, dx) in [(1, 3), (2, 3), (3, 2), (2, 2)]
                dydx = dy * dx
                transformation = (a) -> reshape(a, dy, dx)

                mA = rand(rng, dy, dx)

                Lx = rand(rng, dx, dx)
                μx, Σx = rand(rng, dx), Lx * Lx'

                qx = MvNormalMeanCovariance(μx, Σx)
                qa = MvNormalMeanCovariance(vec(mA), diageye(dydx))
                qW = Wishart(dy + 1, diageye(dy))

                @test_message_update_rule(
                    node = ContinuousTransition, target = :y, algorithm = CTVMP(transformation), atol = 1.0e-5,
                    cases = [(m = (x = qx,), q = (a = qa, W = qW)) => benchmark_rule(qx, qW, mA)],
                )
            end
        end
    end

    @testset "Nonlinear transformation" begin
        @testset "Structured: (m_x::MultivariateNormalDistributionsFamily, q_a::Any, q_W::Any, algo::CTVMP)" begin
            dy, dx = 2, 2
            transformation = (a) -> [cos(a[1]) -sin(a[1]); sin(a[1]) cos(a[1])]

            μx, Σx = zeros(dx), diageye(dx)

            qx = MvNormalMeanCovariance(μx, Σx)
            qa = MvNormalMeanCovariance(zeros(1), tiny * diageye(1))
            qW = Wishart(dy + 1, diageye(dy))

            @test_message_update_rule(
                node = ContinuousTransition, target = :y, algorithm = CTVMP(transformation),
                cases = [(m = (x = qx,), q = (a = qa, W = qW)) => MvGaussianMeanCovariance(zeros(dy), 4 / 3 * diageye(dy))],
            )
        end
    end

    @testset "Mean-field: (q_x::Any, q_a::Any, q_W::Any, algo::CTVMP)" begin
        for (dy, dx) in [(1, 3), (2, 3), (3, 2), (2, 2)]
            dydx = dy * dx
            transformation = (a) -> reshape(a, dy, dx)

            mA = rand(rng, dy, dx)

            Lx = rand(rng, dx, dx)
            μx, Σx = rand(rng, dx), Lx * Lx'

            qx = MvNormalMeanCovariance(μx, Σx)
            qa = MvNormalMeanCovariance(vec(mA), diageye(dydx))
            qW = Wishart(dy + 1, diageye(dy))

            @test_message_update_rule(
                node = ContinuousTransition, target = :y, algorithm = CTVMP(transformation), atol = 1.0e-5,
                cases = [(q = (x = qx, a = qa, W = qW),) => MvNormalMeanPrecision(mA * μx, mean(qW))],
            )
        end
    end
end
