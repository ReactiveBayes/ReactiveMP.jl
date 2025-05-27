
@testitem "rules:ContinuousTransition:W" begin
    using Test, ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions, LinearAlgebra

    import ReactiveMP: @test_rules, ctcompanion_matrix, getjacobians, WishartFast

    rng = MersenneTwister(42)

    @testset "Linear transformation" begin
        # the following rule is used for testing purposes only
        # It is derived separately by Thijs van de Laar
        function benchmark_rule_structured(q_y_x, mA, ΣA, UA)
            myx, Vyx = mean_cov(q_y_x)

            dy  = size(mA, 1)
            Vx  = Vyx[(dy + 1):end, (dy + 1):end]
            Vy  = Vyx[1:dy, 1:dy]
            mx  = myx[(dy + 1):end]
            my  = myx[1:dy]
            Vyx = Vyx[1:dy, (dy + 1):end]

            G = tr(Vx * UA) * ΣA + mA * Vx * mA' - mA * Vyx' - Vyx * mA' + Vy + ΣA * mx' * UA * mx + (mA * mx - my) * (mA * mx - my)'

            return WishartFast(dy + 2, Matrix(Symmetric(G)))
        end

        @testset "Structured: (q_y_x::MultivariateNormalDistributionsFamily, q_a::MultivariateNormalDistributionsFamily, meta::CTMeta)" begin
            for (dy, dx) in [(1, 3), (2, 3), (3, 2), (2, 2)]
                dydx = dy * dx
                transformation = (a) -> reshape(a, dy, dx)
                mA, ΣA, UA = rand(rng, dy, dx), diageye(dy), diageye(dx)

                metal = CTMeta(transformation)
                Lx, Ly = rand(rng, dx, dx), rand(rng, dy, dy)
                μx, Σx = rand(rng, dx), Lx * Lx'
                μy, Σy = rand(rng, dy), Ly * Ly'

                qyx = MvNormalMeanCovariance([μy; μx], [Σy zeros(dy, dx); zeros(dx, dy) Σx])
                qa = MvNormalMeanCovariance(vec(mA), kron(UA, ΣA))

                @test_rules [check_type_promotion = true, atol = 1e-5] ContinuousTransition(:W, Marginalisation) [(
                    input = (q_y_x = qyx, q_a = qa, meta = metal), output = benchmark_rule_structured(qyx, mA, ΣA, UA)
                )]
            end
        end
    end

    @testset "Nonlinear transformation" begin
        @testset "Structured: (q_y_x::MultivariateNormalDistributionsFamily, q_a::Any, q_W::Any, meta::CTMeta)" begin
            dy, dx = 2, 2
            dydx = dy * dy
            transformation = (a) -> [cos(a[1]) -sin(a[1]); sin(a[1]) cos(a[1])]

            metanl = CTMeta(transformation)
            μx, Σx = zeros(dx), diageye(dx)
            μy, Σy = zeros(dy), diageye(dy)

            qyx = MvNormalMeanCovariance([μy; μx], [Σy zeros(dy, dx); zeros(dx, dy) Σx])
            qa = MvNormalMeanCovariance(zeros(1), diageye(1))
            @test_rules [check_type_promotion = true] ContinuousTransition(:W, Marginalisation) [(
                input = (q_y_x = qyx, q_a = qa, meta = metanl), output = WishartFast(dy + 2, dy * diageye(dy))
            )]
        end
    end

    # the following rule is used for testing purposes only
    # It is derived separately by Thijs van de Laar
    function benchmark_rule_meanfield(q_y, q_x, mA, ΣA, UA)
        my, Vy = mean_cov(q_y)
        mx, Vx = mean_cov(q_x)

        dy = size(mA, 1)

        G = tr(Vx * UA) * ΣA + mA * Vx * mA' + Vy + ΣA * mx' * UA * mx + (mA * mx - my) * (mA * mx - my)'

        return WishartFast(dy + 2, Matrix(Symmetric(G)))
    end

    @testset "Mean-field: (q_y::Any, q_x::Any, q_a::Any, meta::CTMeta)" begin
        for (dy, dx) in [(1, 3), (2, 3), (3, 2), (2, 2)]
            dydx = dy * dx
            transformation = (a) -> reshape(a, dy, dx)
            mA, ΣA, UA = rand(rng, dy, dx), diageye(dy), diageye(dx)

            metal = CTMeta(transformation)
            Lx, Ly = rand(rng, dx, dx), rand(rng, dy, dy)
            μx, Σx = rand(rng, dx), (Lx * Lx' + 2 * dy * dx * I)
            μy, Σy = rand(rng, dy), (Ly * Ly' + 2 * dy * dx * I)

            qy = MvNormalMeanCovariance(μy, Σy)
            qx = MvNormalMeanCovariance(μx, Σx)

            qa = MvNormalMeanCovariance(vec(mA), kron(UA, ΣA))

            @test_rules [check_type_promotion = true, atol = 1e-5] ContinuousTransition(:W, Marginalisation) [(
                input = (q_y = qy, q_x = qx, q_a = qa, meta = metal), output = benchmark_rule_meanfield(qy, qx, mA, ΣA, UA)
            )]
        end
    end
end
