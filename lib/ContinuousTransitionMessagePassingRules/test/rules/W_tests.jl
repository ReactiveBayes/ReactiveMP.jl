@testitem "rules:ContinuousTransition:W" tags = [:rules] begin
    using ContinuousTransitionMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, LinearAlgebra, Random
    using ExponentialFamily: WishartFast

    diageye(n) = Matrix{Float64}(I, n, n)

    rng = MersenneTwister(42)

    @testset "Linear transformation" begin
        # the following rule is used for testing purposes only
        # It is derived separately by Thijs van de Laar
        function benchmark_rule_structured(q_y_x, mA, ΣA, UA)
            myx, Vyx = mean_cov(q_y_x)

            dy = size(mA, 1)
            Vx = Vyx[(dy + 1):end, (dy + 1):end]
            Vy = Vyx[1:dy, 1:dy]
            mx = myx[(dy + 1):end]
            my = myx[1:dy]
            Vyx = Vyx[1:dy, (dy + 1):end]

            G =
                tr(Vx * UA) * ΣA + mA * Vx * mA' - mA * Vyx' - Vyx * mA' +
                Vy +
                ΣA * mx' * UA * mx +
                (mA * mx - my) * (mA * mx - my)'

            return WishartFast(dy + 2, Matrix(Symmetric(G)))
        end

        @testset "Structured: (q_y_x::MultivariateNormalDistributionsFamily, q_a::MultivariateNormalDistributionsFamily, algo::CTVMP)" begin
            for (dy, dx) in [(1, 3), (2, 3), (3, 2), (2, 2)]
                transformation = (a) -> reshape(a, dy, dx)
                mA, ΣA, UA = rand(rng, dy, dx), diageye(dy), diageye(dx)

                Lx, Ly = rand(rng, dx, dx), rand(rng, dy, dy)
                μx, Σx = rand(rng, dx), Lx * Lx'
                μy, Σy = rand(rng, dy), Ly * Ly'

                qyx = MvNormalMeanCovariance([μy; μx], [Σy zeros(dy, dx); zeros(dx, dy) Σx])
                qa = MvNormalMeanCovariance(vec(mA), kron(UA, ΣA))

                @test_message_update_rule(
                    node = ContinuousTransition, target = :W, algorithm = CTVMP(transformation), atol = 1.0e-5,
                    cases = [(clusters = ((:y, :x) => qyx,), q = (a = qa,)) => benchmark_rule_structured(qyx, mA, ΣA, UA)],
                )
            end
        end
    end

    # the following rule is used for testing purposes only
    # It is derived separately by Thijs van de Laar
    function benchmark_rule_meanfield(q_y, q_x, mA, ΣA, UA)
        my, Vy = mean_cov(q_y)
        mx, Vx = mean_cov(q_x)

        dy = size(mA, 1)

        G =
            tr(Vx * UA) * ΣA +
            mA * Vx * mA' +
            Vy +
            ΣA * mx' * UA * mx +
            (mA * mx - my) * (mA * mx - my)'

        return WishartFast(dy + 2, Matrix(Symmetric(G)))
    end

    @testset "Mean-field: (q_y::Any, q_x::Any, q_a::Any, algo::CTVMP)" begin
        for (dy, dx) in [(1, 3), (2, 3), (3, 2), (2, 2)]
            transformation = (a) -> reshape(a, dy, dx)
            mA, ΣA, UA = rand(rng, dy, dx), diageye(dy), diageye(dx)

            Lx, Ly = rand(rng, dx, dx), rand(rng, dy, dy)
            μx, Σx = rand(rng, dx), Lx * Lx'
            μy, Σy = rand(rng, dy), Ly * Ly'

            qy = MvNormalMeanCovariance(μy, Σy)
            qx = MvNormalMeanCovariance(μx, Σx)

            qa = MvNormalMeanCovariance(vec(mA), kron(UA, ΣA))

            @test_message_update_rule(
                node = ContinuousTransition, target = :W, algorithm = CTVMP(transformation), atol = 1.0e-5,
                cases = [(q = (y = qy, x = qx, a = qa),) => benchmark_rule_meanfield(qy, qx, mA, ΣA, UA)],
            )
        end
    end
end

# A rotation: the rule computes E[(y - A x)(y - A x)ᵀ] with the offset f(m_a) - J m_a of a
# nonlinear f, so the value is not the one rows of A linear in `a` through the origin would give.
# The offset is tested exactly in `offset_tests.jl`; this item only asserts a proper result.
@testitem "rules:ContinuousTransition:W, a nonlinear transformation" tags = [:rules] begin
    using ContinuousTransitionMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, LinearAlgebra
    using ExponentialFamily: WishartFast

    diageye(n) = Matrix{Float64}(I, n, n)

    @testset "Structured: (q_y_x::MultivariateNormalDistributionsFamily, q_a::Any, algo::CTVMP)" begin
        dy, dx = 2, 2
        transformation = (a) -> [cos(a[1]) -sin(a[1]); sin(a[1]) cos(a[1])]

        μx, Σx = zeros(dx), diageye(dx)
        μy, Σy = zeros(dy), diageye(dy)

        qyx = MvNormalMeanCovariance([μy; μx], [Σy zeros(dy, dx); zeros(dx, dy) Σx])
        qa = MvNormalMeanCovariance(zeros(1), diageye(1))
        result = getresult(call_message_update_rule(ContinuousTransition, :W; clusters = ((:y, :x) => qyx,), q = (a = qa,), algorithm = CTVMP(transformation)))
        @test result isa WishartFast
        ν, invS = params(result)
        @test ν == dy + 2
        @test size(invS) == (dy, dy) && all(isfinite, invS) && isposdef(Matrix(invS))
    end
end
