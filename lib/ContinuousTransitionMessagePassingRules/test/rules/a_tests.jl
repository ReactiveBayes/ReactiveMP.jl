# From v6's `test/rules/continuous_transition/a_tests.jl`.

@testitem "rules:ContinuousTransition:a" tags = [:rules] begin
    using ContinuousTransitionMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, LinearAlgebra, Random

    diageye(n) = Matrix{Float64}(I, n, n)

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
                Lx, Ly = rand(rng, dx, dx), rand(rng, dy, dy)
                μx, Σx = rand(rng, dx), Lx * Lx'
                μy, Σy = rand(rng, dy), Ly * Ly'

                qyx = MvNormalMeanCovariance([μy; μx], [Σy zeros(dy, dx); zeros(dx, dy) Σx])
                qa = MvNormalMeanCovariance(a0, diageye(dydx))
                qW = Wishart(dy + 1, diageye(dy))
                @test_message_update_rule(
                    node = ContinuousTransition, target = :a, algorithm = CTVMP(transformation), check_type_promotion = false,
                    cases = [(clusters = ((:y, :x) => qyx,), q = (a = qa, W = qW)) => benchmark_rule_structured(qyx, qW)],
                )
            end
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
            Lx, Ly = rand(rng, dx, dx), rand(rng, dy, dy)
            μx, Σx = rand(rng, dx), Lx * Lx'
            μy, Σy = rand(rng, dy), Ly * Ly'
            qy = MvNormalMeanCovariance(μy, Σy)
            qx = MvNormalMeanCovariance(μx, Σx)
            qa = MvNormalMeanCovariance(a0, diageye(dydx))
            qW = Wishart(dy + 1, diageye(dy))
            @test_message_update_rule(
                node = ContinuousTransition, target = :a, algorithm = CTVMP(transformation), check_type_promotion = false,
                cases = [
                    (q = (y = qy, x = qx, a = qa, W = qW),) => benchmark_rule_meanfield(qy, qx, qW),
                    (q = (y = PointMass(μy), x = qx, a = qa, W = qW),) => benchmark_rule_meanfield(PointMass(μy), qx, qW),
                ],
            )
        end
    end
end

# v6's rotation case, whose value v6 took from rows of A linear in `a` through the origin. The
# port keeps the offset f(m_a) - J m_a of a nonlinear f, so its value differs from v6's
# MvNormalWeightedMeanPrecision(zeros(1), 8 * diageye(1)); the exact tests of the correction are
# elsewhere, and this one only asserts a proper result.
@testitem "rules:ContinuousTransition:a, v6's nonlinear cases (corrected)" tags = [:rules] begin
    using ContinuousTransitionMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, LinearAlgebra

    diageye(n) = Matrix{Float64}(I, n, n)

    @testset "Structured: (q_y_x::MultivariateNormalDistributionsFamily, q_a::Any, q_W::Any, meta::CTMeta)" begin
        dy, dx = 2, 2
        transformation = (a) -> [cos(a[1]) -sin(a[1]); sin(a[1]) cos(a[1])]
        a0 = zeros(Int, 1)
        μx, Σx = ones(dx), diageye(dx)
        μy, Σy = ones(dy), diageye(dy)

        qyx = MvNormalMeanCovariance([μy; μx], [Σy zeros(dy, dx); zeros(dx, dy) Σx])
        qa = MvNormalMeanCovariance(a0, diageye(1))
        qW = Wishart(dy, diageye(dy))
        result = call_message_update_rule(ContinuousTransition, :a; clusters = ((:y, :x) => qyx,), q = (a = qa, W = qW), algorithm = CTVMP(transformation))
        @test result isa MvNormalWeightedMeanPrecision{Float64}
        ξ, Λ = weightedmean_precision(result)
        @test length(ξ) == 1 && all(isfinite, ξ)
        @test all(isfinite, Λ) && isposdef(Λ)
    end
end
