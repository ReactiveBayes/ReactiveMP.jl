@testitem "rules:ContinuousTransition:y" tags = [:rules] begin
    using ContinuousTransitionMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, LinearAlgebra, Random
    using BayesBase: tiny

    diageye(n) = Matrix{Float64}(I, n, n)

    rng = MersenneTwister(42)

    @testset "Linear transformation" begin
        # ∫ m_x(x) exp⟨log N(y; A x, W⁻¹)⟩ dx with a = vec(A): E[Aᵢₖ Aⱼₗ] = mAᵢₖ mAⱼₗ + Cov(Aᵢₖ, Aⱼₗ),
        # so the uncertainty of A adds K = E[AᵀWA] - mAᵀ⟨W⟩mA to the precision of x.
        function benchmark_rule(q_x, q_a, q_W, mA)
            dy, dx = size(mA)
            ξx, Wx = weightedmean_precision(q_x)
            Va, mW = cov(q_a), mean(q_W)
            K = [sum(mW[i, j] * Va[(k - 1) * dy + i, (l - 1) * dy + j] for i in 1:dy, j in 1:dy) for k in 1:dx, l in 1:dx]
            Vx = inv(Wx + K)
            return MvNormalMeanCovariance(mA * Vx * ξx, mA * Vx * mA' + inv(mW))
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

                # with `a` known, the message is the transition of m_x, N(mA mx, mA Vx mAᵀ + ⟨W⟩⁻¹)
                qa_known = MvNormalMeanCovariance(vec(mA), tiny * diageye(dydx))

                @test_message_update_rule(
                    node = ContinuousTransition, target = :y, algorithm = CTVMP(transformation), atol = 1.0e-5,
                    cases = [
                        (m = (x = qx,), q = (a = qa, W = qW)) => benchmark_rule(qx, qa, qW, mA),
                        (m = (x = qx,), q = (a = qa_known, W = qW)) => MvNormalMeanCovariance(mA * μx, mA * Σx * mA' + inv(mean(qW))),
                    ],
                )
            end
        end
    end

    # The joint q(y, x) is m_y(y) m_x(x) exp⟨log N(y; A x, W⁻¹)⟩, and the message towards `y` is
    # that factor with `x` integrated out, so m_y times the message is the joint's q(y).
    @testset "the message agrees with the joint marginal" begin
        rotation(a) = [cos(a[1]) -sin(a[1]); sin(a[1]) cos(a[1])] * a[2]
        for (transformation, da) in ((a -> reshape(a, 2, 2), 4), (rotation, 2))
            algorithm = CTVMP(transformation)
            L = rand(rng, da, da)
            m_y = MvNormalMeanCovariance(randn(rng, 2), [0.6 0.1; 0.1 0.4])
            m_x = MvNormalMeanCovariance(randn(rng, 2), [1.0 0.2; 0.2 0.7])
            q_a = MvNormalMeanCovariance(rand(rng, da), L * L' / da + 0.1 * diageye(da))
            q_W = Wishart(4.0, [0.5 0.1; 0.1 0.3])
            message = getresult(call_message_update_rule(ContinuousTransition, :y; m = (x = m_x,), q = (a = q_a, W = q_W), algorithm))
            joint = getresult(call_marginal_update_rule(ContinuousTransition, (:y, :x); m = (y = m_y, x = m_x), q = (a = q_a, W = q_W), algorithm))
            q_y = prod(GenericProd(), m_y, message)
            @test mean(q_y) ≈ mean(joint)[1:2] rtol = 1.0e-10
            @test cov(q_y) ≈ cov(joint)[1:2, 1:2] rtol = 1.0e-10
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
