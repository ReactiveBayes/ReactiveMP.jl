
@testitem "rules:ContinuousTransition:y" begin
    using Test, ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules, ctcompanion_matrix, getjacobians

    rng = MersenneTwister(42)

    @testset "Linear transformation" begin
        # the following rule is used for testing purposes only
        # It is derived separately by Thijs van de Laar
        function benchmark_rule(q_x, q_W, mA)
            mx, Vx = mean_cov(q_x)
            mW = mean(q_W)
            return MvNormalMeanCovariance(mA * mx, mA * Vx * mA' + inv(mW))
        end

        @testset "Structured: (m_x::MultivariateNormalDistributionsFamily, q_a::MultivariateNormalDistributionsFamily, q_W::Any, meta::CTMeta)" begin
            for (dy, dx) in [(1, 3), (2, 3), (3, 2), (2, 2)]
                dydx = dy * dx
                transformation = (a) -> reshape(a, dy, dx)

                mA = rand(rng, dy, dx)

                metal = CTMeta(transformation)
                Lx = rand(rng, dx, dx)
                μx, Σx = rand(rng, dx), Lx * Lx'

                qx = MvNormalMeanCovariance(μx, Σx)
                qa = MvNormalMeanCovariance(vec(mA), diageye(dydx))
                qW = Wishart(dy + 1, diageye(dy))

                @test_rules [check_type_promotion = true, atol = 1e-5] ContinuousTransition(:y, Marginalisation) [(
                    input = (m_x = qx, q_a = qa, q_W = qW, meta = metal), output = benchmark_rule(qx, qW, mA)
                )]
            end
        end
    end

    @testset "Nonlinear transformation" begin
        @testset "Structured: (m_x::MultivariateNormalDistributionsFamily, q_a::Any, q_W::Any, meta::CTMeta)" begin
            dy, dx = 2, 2
            dydx = dy * dy
            transformation = (a) -> [cos(a[1]) -sin(a[1]); sin(a[1]) cos(a[1])]

            metanl = CTMeta(transformation)
            μx, Σx = zeros(dx), diageye(dx)

            qx = MvNormalMeanCovariance(μx, Σx)
            qa = MvNormalMeanCovariance(zeros(1), tiny * diageye(1))
            qW = Wishart(dy + 1, diageye(dy))

            @test_rules [check_type_promotion = true] ContinuousTransition(:y, Marginalisation) [(
                input = (m_x = qx, q_a = qa, q_W = qW, meta = metanl), output = MvGaussianMeanCovariance(zeros(dy), 4 / 3 * diageye(dy))
            )]
        end
    end

    @testset "Mean-field: (q_y::Any, q_a::Any, q_W::Any, meta::CTMeta)" begin
        for (dy, dx) in [(1, 3), (2, 3), (3, 2), (2, 2)]
            dydx = dy * dx
            transformation = (a) -> reshape(a, dy, dx)

            mA = rand(rng, dy, dx)

            metal = CTMeta(transformation)
            Lx = rand(rng, dx, dx)
            μx, Σx = rand(rng, dx), Lx * Lx'

            qx = MvNormalMeanCovariance(μx, Σx)
            qa = MvNormalMeanCovariance(vec(mA), diageye(dydx))
            qW = Wishart(dy + 1, diageye(dy))

            @test_rules [check_type_promotion = true, atol = 1e-5] ContinuousTransition(:y, Marginalisation) [(
                input = (q_x = qx, q_a = qa, q_W = qW, meta = metal), output = MvNormalMeanPrecision(mA * μx, mean(qW))
            )]
        end
    end
end
