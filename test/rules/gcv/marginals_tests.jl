
@testitem "marginalrules:GCV:y_x" setup = [GCVRulesTestUtils] begin
    using ReactiveMP, BayesBase, ExponentialFamily, Distributions, LinearAlgebra

    import ReactiveMP:
        @test_marginalrules, ExponentialLinearQuadratic, GaussHermiteCubature

    expected_A     = GCVRulesTestUtils.expected_A
    expected_B     = GCVRulesTestUtils.expected_B
    default_meta   = GCVRulesTestUtils.default_meta
    parameter_sets = GCVRulesTestUtils.parameter_sets

    # The joint `q(y, x)` combines the two incoming messages with the factor
    # `N(y | x, exp(κz + ω))`. Writing the factor's exponent in terms of `(y, x)` with the
    # effective noise precision `AB = ⟨e^{-(κz+ω)}⟩`:
    #
    #     -½·AB·(y - x)²  ⇒  precision contribution  [ AB  -AB ; -AB  AB ]
    #
    # Adding the incoming messages' own precisions on the diagonal, and their weighted means
    # as the natural parameter:
    #
    #     W = [ W_y + AB   -AB      ]        ξ = [ ⟨y⟩·W_y ]
    #         [ -AB        W_x + AB ]            [ ⟨x⟩·W_x ]
    #
    # The off-diagonal is the *only* place the two variables couple, and it is exactly the
    # negated noise precision -- which is also the reciprocal of the variance the `:y`/`:x`
    # rules add. That cross-consistency is asserted below.
    meta = default_meta()

    @testset "Against the derived weighted-mean/precision form" begin
        for params in parameter_sets()
            (; q_y, q_x, q_z, q_κ, q_ω) = params
            m_y, w_y = mean_precision(q_y)
            m_x, w_x = mean_precision(q_x)
            ab = expected_A(q_ω) * expected_B(q_z, q_κ)

            result = @call_marginalrule GCV(:y_x) (
                m_y = q_y,
                m_x = q_x,
                q_z = q_z,
                q_κ = q_κ,
                q_ω = q_ω,
                meta = meta,
            )

            @test result isa MvNormalWeightedMeanPrecision
            @test weightedmean(result) ≈ [m_y * w_y, m_x * w_x]
            @test invcov(result) ≈ [w_y+ab -ab; -ab w_x+ab]
        end
    end

    @testset "Off-diagonal coupling is exactly the noise precision" begin
        # Ties the marginal rule to the `:y` rule: the joint's off-diagonal is `-AB`, and the
        # variance the `:y` rule adds on top of the incoming variance is `1/AB`. Two different
        # code paths computing the same effective noise, checked against each other rather than
        # against a shared reference value.
        for params in parameter_sets()
            (; q_y, q_x, q_z, q_κ, q_ω) = params

            joint = @call_marginalrule GCV(:y_x) (
                m_y = q_y,
                m_x = q_x,
                q_z = q_z,
                q_κ = q_κ,
                q_ω = q_ω,
                meta = meta,
            )
            to_y = @call_rule GCV(:y, Marginalisation) (
                m_x = q_x, q_z = q_z, q_κ = q_κ, q_ω = q_ω, meta = meta
            )

            W = invcov(joint)
            added_variance = var(to_y) - var(q_x)

            @test W[1, 2] ≈ W[2, 1]
            @test -W[1, 2] ≈ inv(added_variance)

            # The coupling also appears on the diagonal, so each diagonal entry minus the
            # corresponding incoming precision must equal the same quantity.
            @test W[1, 1] - precision(q_y) ≈ -W[1, 2]
            @test W[2, 2] - precision(q_x) ≈ -W[1, 2]
        end
    end

    @testset "The joint is a valid, positive-definite Gaussian" begin
        # `W` is a rank-one coupling added to a positive diagonal, so it must stay PD -- a
        # sanity property that catches sign errors in the off-diagonal.
        for params in parameter_sets()
            (; q_y, q_x, q_z, q_κ, q_ω) = params

            joint = @call_marginalrule GCV(:y_x) (
                m_y = q_y,
                m_x = q_x,
                q_z = q_z,
                q_κ = q_κ,
                q_ω = q_ω,
                meta = meta,
            )

            @test isposdef(invcov(joint))
            @test isposdef(cov(joint))
            # Positive coupling between y and x: observing one informs the other upward.
            @test cov(joint)[1, 2] > 0
        end
    end

    @testset "Marginal means bracket the two incoming means" begin
        # With a shared noise term pulling `y` and `x` together, each marginal mean must lie
        # between the two incoming means (inclusive), never outside them.
        for params in parameter_sets()
            (; q_y, q_x, q_z, q_κ, q_ω) = params

            joint = @call_marginalrule GCV(:y_x) (
                m_y = q_y,
                m_x = q_x,
                q_z = q_z,
                q_κ = q_κ,
                q_ω = q_ω,
                meta = meta,
            )

            m = mean(joint)
            lo, hi = minmax(mean(q_y), mean(q_x))
            @test lo - 1e-10 <= m[1] <= hi + 1e-10
            @test lo - 1e-10 <= m[2] <= hi + 1e-10
        end
    end

    @testset "Type promotion, against fully worked-out constants" begin
        # A·B = exp(-0.936) as derived in `y_tests.jl`. With m_y = N(3.0, 1.0) and
        # m_x = N(1.0, 2.0): W_y = 1.0, W_x = 0.5, ξ = [3.0, 0.5].
        ab = exp(-0.936)
        @test_marginalrules [check_type_promotion = true] GCV(
            :y_x
        ) [(
            input = (
                m_y = NormalMeanVariance(3.0, 1.0),
                m_x = NormalMeanVariance(1.0, 2.0),
                q_z = NormalMeanVariance(0.5, 0.7),
                q_κ = NormalMeanVariance(0.8, 0.4),
                q_ω = NormalMeanVariance(1.2, 0.5),
                meta = GCVMetadata(GaussHermiteCubature(20)),
            ),
            output = MvNormalWeightedMeanPrecision(
                [3.0, 0.5], [1.0+ab -ab; -ab 0.5+ab]
            ),
        )]
    end

    @testset "Accepts ExponentialLinearQuadratic incoming messages" begin
        # Both `m_y` and `m_x` are typed `UniNormalOrExpLinQuad`.
        (; q_x, q_z, q_κ, q_ω) = first(parameter_sets())
        elq = ExponentialLinearQuadratic(
            GaussHermiteCubature(20), 1.0, 1.0, -1.0, 0.0
        )
        elq_mean, elq_precision = mean_precision(elq)
        ab = expected_A(q_ω) * expected_B(q_z, q_κ)
        m_x, w_x = mean_precision(q_x)

        joint = @call_marginalrule GCV(:y_x) (
            m_y = elq,
            m_x = q_x,
            q_z = q_z,
            q_κ = q_κ,
            q_ω = q_ω,
            meta = meta,
        )

        @test joint isa MvNormalWeightedMeanPrecision
        @test weightedmean(joint) ≈ [elq_mean * elq_precision, m_x * w_x]
        @test invcov(joint) ≈ [elq_precision+ab -ab; -ab w_x+ab]
    end
end
