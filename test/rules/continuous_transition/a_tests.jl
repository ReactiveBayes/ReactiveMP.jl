
@testitem "rules:ContinuousTransition:a" begin
    using Test, ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules, getjacobians

    rng = MersenneTwister(42)

    @testset "Linear transformation" begin

        # the following rule is used for testing purposes only
        # It is derived separately by Thijs van de Laar
        function benchmark_rule_structured(q_y_x, q_W)
            myx, Vyx = mean_cov(q_y_x)
            dy = size(q_W.S, 1)
            Vx = Vyx[(dy + 1):end, (dy + 1):end]
            mx = myx[(dy + 1):end]
            my = myx[1:dy]
            Vyx = Vyx[1:dy, (dy + 1):end]
            mW = mean(q_W)
            Λ = kron(Vx + mx * mx', mW)
            return MvNormalWeightedMeanPrecision(Λ * vec((Vyx + my * mx') * inv((Vx + mx * mx'))), Λ)
        end

        @testset "Structured: (q_y_x::MultivariateNormalDistributionsFamily, q_a::MultivariateNormalDistributionsFamily, q_W::Any, meta::CTMeta)" begin
            for (dy, dx) in [(1, 3), (2, 3), (3, 2), (2, 2)]
                dydx = dy * dx
                transformation = (a) -> reshape(a, dy, dx)
                a0 = rand(Float32, dydx)
                metal = CTMeta(transformation)
                Lx, Ly = rand(rng, dx, dx), rand(rng, dy, dy)
                μx, Σx = rand(rng, dx), Lx * Lx'
                μy, Σy = rand(rng, dy), Ly * Ly'

                qyx = MvNormalMeanCovariance([μy; μx], [Σy zeros(dy, dx); zeros(dx, dy) Σx])
                qa = MvNormalMeanCovariance(a0, diageye(dydx))
                qW = Wishart(dy + 1, diageye(dy))
                @test_rules [check_type_promotion = false] ContinuousTransition(:a, Marginalisation) [(
                    input = (q_y_x = qyx, q_a = qa, q_W = qW, meta = metal), output = benchmark_rule_structured(qyx, qW)
                )]
            end
        end
    end

    @testset "Nonlinear transformation" begin
        @testset "Structured: (q_y_x::MultivariateNormalDistributionsFamily, q_a::Any, q_W::Any, meta::CTMeta)" begin
            dy, dx = 2, 2
            dydx = dy * dy
            transformation = (a) -> [cos(a[1]) -sin(a[1]); sin(a[1]) cos(a[1])]
            a0 = zeros(Int, 1)
            metanl = CTMeta(transformation)
            μx, Σx = ones(dx), diageye(dx)
            μy, Σy = ones(dy), diageye(dy)

            qyx = MvNormalMeanCovariance([μy; μx], [Σy zeros(dy, dx); zeros(dx, dy) Σx])
            qa = MvNormalMeanCovariance(a0, diageye(1))
            qW = Wishart(dy, diageye(dy))
            @test_rules [check_type_promotion = true] ContinuousTransition(:a, Marginalisation) [(
                input = (q_y_x = qyx, q_a = qa, q_W = qW, meta = metanl), output = MvNormalWeightedMeanPrecision(zeros(1), (qW.df * dy * dx) * diageye(1))
            )]
        end
    end

    # the following rule is used for testing purposes only
    # It is derived separately by Thijs van de Laar
    # NOTE: this test rule does not allow q_x as a PointMass as it involves the covariance matrix of q_x
    function benchmark_rule_meanfield(q_y, q_x, q_W)
        my = mean(q_y)
        mx, Vx = mean_cov(q_x)
        mW = mean(q_W)
        Λ = kron(Vx + mx * mx', mW)
        return MvNormalWeightedMeanPrecision(Λ * (vec(my * mx' * inv(Vx + mx * mx'))), Λ)
    end

    @testset "Mean-field: (q_y::Any, q_x::Any, q_a::Any, q_W::Any, meta::CTMeta)" begin
        for (dy, dx) in [(1, 3), (2, 3), (3, 2), (2, 2)]
            dydx = dy * dx
            transformation = (a) -> reshape(a, dy, dx)
            a0 = rand(Float32, dydx)
            metal = CTMeta(transformation)
            Lx, Ly = rand(rng, dx, dx), rand(rng, dy, dy)
            μx, Σx = rand(rng, dx), Lx * Lx'
            μy, Σy = rand(rng, dy), Ly * Ly'
            qy = MvNormalMeanCovariance(μy, Σy)
            qx = MvNormalMeanCovariance(μx, Σx)
            qa = MvNormalMeanCovariance(a0, diageye(dydx))
            qW = Wishart(dy + 1, diageye(dy))
            @test_rules [check_type_promotion = false] ContinuousTransition(:a, Marginalisation) [(
                input = (q_y = qy, q_x = qx, q_a = qa, q_W = qW, meta = metal), output = benchmark_rule_meanfield(qy, qx, qW)
            )]

            @test_rules [check_type_promotion = false] ContinuousTransition(:a, Marginalisation) [(
                input = (q_y = PointMass(μy), q_x = qx, q_a = qa, q_W = qW, meta = metal), output = benchmark_rule_meanfield(PointMass(μy), qx, qW)
            )]
        end
    end
end
