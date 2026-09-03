@testitem "rules:GaussianCoupling:marginals" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions
    using LinearAlgebra

    import ReactiveMP: @test_marginalrules

    @testset "out_in: (m_out::UnivariateNormalDistributionsFamily, m_in::UnivariateNormalDistributionsFamily, q_a::PointMass)" begin
        # q(out, in) ∝ m_out(out) m_in(in) exp(a⋅out⋅in), so in weighted-mean/precision
        # form ξ = [ξ_out, ξ_in] and W = [w_out -a; -a w_in]: the factor contributes the
        # cross-term only, never the quadratic self-terms.
        @test_marginalrules [check_type_promotion = true] GaussianCoupling(
            :out_in
        ) [
            (
                input  = (m_out = NormalWeightedMeanPrecision(1.0, 2.0), m_in = NormalWeightedMeanPrecision(2.0, 3.0), q_a = PointMass(1.0)),
                output = MvNormalWeightedMeanPrecision([1.0, 2.0], [2.0 -1.0; -1.0 3.0]),
            ),
            # Negative coupling — this is the actual usage, since a = -A[i, j].
            (
                input  = (m_out = NormalWeightedMeanPrecision(1.0, 2.0), m_in = NormalWeightedMeanPrecision(2.0, 3.0), q_a = PointMass(-1.5)),
                output = MvNormalWeightedMeanPrecision([1.0, 2.0], [2.0 1.5; 1.5 3.0]),
            ),
            (
                input  = (m_out = NormalMeanVariance(1.0, 0.5), m_in = NormalMeanVariance(-2.0, 0.25), q_a = PointMass(-1.5)),
                output = MvNormalWeightedMeanPrecision([2.0, -8.0], [2.0 1.5; 1.5 4.0]),
            ),
            (
                input  = (m_out = NormalMeanPrecision(2.0, 4.0), m_in = NormalMeanPrecision(-1.0, 2.0), q_a = PointMass(0.5)),
                output = MvNormalWeightedMeanPrecision([8.0, -2.0], [4.0 -0.5; -0.5 2.0]),
            ),
            # A zero coupling leaves the joint block-diagonal.
            (
                input  = (m_out = NormalMeanVariance(1.0, 0.5), m_in = NormalMeanVariance(-2.0, 0.25), q_a = PointMass(0.0)),
                output = MvNormalWeightedMeanPrecision([2.0, -8.0], [2.0 0.0; 0.0 4.0]),
            ),
        ]
    end

    @testset "Properness boundary" begin
        # The joint is proper iff precision(m_out) ⋅ precision(m_in) > mean(q_a)², i.e.
        # exactly when the incoming precisions dominate the coupling. Note that this is a
        # *local* condition on a single factor, not a convergence condition on the model.
        for (w_out, w_in, a, isproper) in (
            (4.0, 4.0, 1.0, true),   # 16 > 1
            (2.0, 2.0, 1.9, true),   # 4 > 3.61
            (2.0, 2.0, 2.0, false),  # 4 == 4, singular
            (1.0, 1.0, 2.0, false),  # 1 < 4, indefinite
        )
            q = @call_marginalrule GaussianCoupling(:out_in) (
                m_out = NormalWeightedMeanPrecision(0.5, w_out),
                m_in  = NormalWeightedMeanPrecision(-0.5, w_in),
                q_a   = PointMass(a),
            )
            # Allow for roundoff in the zero eigenvalue at the exactly singular boundary.
            W = Symmetric(convert(Matrix, precision(q)))
            tolerance = sqrt(eps(eltype(W))) * opnorm(W)
            @test (eigmin(W) > tolerance) === isproper
            @test (w_out * w_in > abs2(a)) === isproper
        end
    end
end
