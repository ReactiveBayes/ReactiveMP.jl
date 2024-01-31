
@testset "rules:ContinuousTransition:x" begin
    using Test, ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions, LinearAlgebra

    import ReactiveMP: @test_rules, ctcompanion_matrix, getjacobians

    rng = MersenneTwister(42)

    @testset "Linear transformation" begin
        # the following rule is used for testing purposes only
        # It is derived separately by Thijs van de Laar
        function benchmark_rule(q_y, q_W, mA, ΣA, UA)
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

                metal = CTMeta(transformation)
                Lx, Ly = rand(rng, dx, dx), rand(rng, dy, dy)
                μy, Σy = rand(rng, dy), Ly * Ly'

                qy = MvNormalMeanCovariance(μy, Σy)
                qa = MvNormalMeanCovariance(vec(mA), diageye(dydx))
                qW = Wishart(dy + 1, diageye(dy))

                @test_rules [check_type_promotion = true, atol = 1e-4] ContinuousTransition(:x, Marginalisation) [(
                    input = (m_y = qy, q_a = qa, q_W = qW, meta = metal), output = benchmark_rule(qy, qW, mA, ΣA, UA)
                )]
            end
        end
    end

    @testset "Nonlinear transformation" begin
        @testset "Structured: (m_y::MultivariateNormalDistributionsFamily, q_a::Any, q_W::Any, meta::CTMeta)" begin
            dy, dx = 2, 2
            dydx = dy * dy
            transformation = (a) -> [cos(a[1]) -sin(a[1]); sin(a[1]) cos(a[1])]

            metanl = CTMeta(transformation)
            μy, Σy = zeros(dy), diageye(dy)

            qy = MvNormalMeanCovariance(μy, Σy)
            qa = MvNormalMeanCovariance(zeros(1), tiny * diageye(1))
            qW = Wishart(dy + 1, diageye(dy))

            @test_rules [check_type_promotion = true] ContinuousTransition(:x, Marginalisation) [(
                input = (m_y = qy, q_a = qa, q_W = qW, meta = metanl), output = MvGaussianWeightedMeanPrecision(zeros(dx), 3 / 4 * diageye(dx))
            )]
        end
    end
end
