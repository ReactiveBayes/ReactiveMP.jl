@testmodule GCVRulesTestUtils begin
    using ReactiveMP, BayesBase, ExponentialFamily, Distributions

    import ReactiveMP:
        ExponentialLinearQuadratic, GCVMetadata, GaussHermiteCubature

    # The `GCV` node is `p(y | x, z, κ, ω) = N(y | x, exp(κz + ω))` -- the *variance* is
    # `exp(κz + ω)`, so the precision is `exp(-(κz + ω))`. Everything in these test files is
    # derived from that single statement, which is pinned independently by the node's own
    # `@average_energy`:
    #
    #     -log p = ½[log2π + (κz + ω) + (y - x)²·e^{-(κz + ω)}]
    #
    # Two expectations recur throughout. Only the first is exact:
    #
    #   A = ⟨e^{-ω}⟩  -- EXACT for ω ~ N(m, v): the mean of a lognormal, exp(-m + v/2).
    #
    #   B = ⟨e^{-κz}⟩ -- an APPROXIMATION. `κz` is a product of two Gaussians and is not
    #                    itself Gaussian, but the node treats it as such and applies the
    #                    same lognormal formula using Var(κz) = ⟨κ⟩²Var(z) + ⟨z⟩²Var(κ) +
    #                    Var(κ)Var(z). This is a deliberate modelling choice of the node,
    #                    not a defect, so the tests below reproduce it rather than
    #                    comparing against the true expectation.
    #
    # `A` is verified against an independent Monte Carlo estimate in `shared_tests.jl`;
    # `B`'s status as an approximation is documented there too.

    # Exact: mean of the lognormal e^{-ω} for ω ~ N(m_ω, v_ω)
    expected_A(q_ω) = ((m, v) = mean_var(q_ω); exp(-m + v / 2))

    # The node's Gaussian-moment approximation to ⟨e^{-κz}⟩
    function expected_B(q_z, q_κ)
        m_z, v_z = mean_var(q_z)
        m_κ, v_κ = mean_var(q_κ)
        var_κz = m_κ^2 * v_z + m_z^2 * v_κ + v_κ * v_z
        return exp(-m_κ * m_z + var_κz / 2)
    end

    # ⟨(y - x)²⟩ under a factorized q(y)q(x)
    expected_psi(q_y, q_x) =
        let (m_y, v_y) = mean_var(q_y), (m_x, v_x) = mean_var(q_x)
            (m_y - m_x)^2 + v_y + v_x
        end

    # ⟨(y - x)²⟩ under a joint q(y, x); equals the factorized form when Cov(y, x) = 0
    function expected_psi(q_y_x)
        m, V = mean_cov(q_y_x)
        return (m[1] - m[2])^2 + V[1, 1] + V[2, 2] - V[1, 2] - V[2, 1]
    end

    coefficients(d::ExponentialLinearQuadratic) = (d.a, d.b, d.c, d.d)

    # A joint q(y, x) with zero cross-covariance, i.e. the factorized case embedded in the
    # structured parameterisation. Every structured rule must then agree exactly with its
    # mean-field sibling.
    block_diagonal_joint(q_y, q_x) =
        let (m_y, v_y) = mean_var(q_y), (m_x, v_x) = mean_var(q_x)
            MvNormalMeanCovariance([m_y, m_x], [v_y 0.0; 0.0 v_x])
        end

    default_meta() = GCVMetadata(GaussHermiteCubature(20))

    # A spread of parameter sets used across the rule test files. `z` and `κ` deliberately
    # get means away from 0 and 1 and non-zero variances -- degenerate choices such as
    # ⟨z⟩ = 1, Var(z) = 0 mask coefficient mix-ups between the `:ω`, `:κ` and `:z` rules.
    function parameter_sets()
        return (
            (
                q_y = NormalMeanVariance(3.0, 1.0),
                q_x = NormalMeanVariance(1.0, 2.0),
                q_z = NormalMeanVariance(0.5, 0.7),
                q_κ = NormalMeanVariance(0.8, 0.4),
                q_ω = NormalMeanVariance(1.2, 0.5),
            ),
            (
                q_y = NormalMeanVariance(0.4, 0.6),
                q_x = NormalMeanVariance(0.5, 0.3),
                q_z = NormalMeanVariance(2.0, 0.3),
                q_κ = NormalMeanVariance(1.2, 0.25),
                q_ω = NormalMeanVariance(-0.5, 0.8),
            ),
            (
                q_y = NormalMeanVariance(-1.5, 0.25),
                q_x = NormalMeanVariance(2.5, 0.5),
                q_z = NormalMeanVariance(-0.75, 1.25),
                q_κ = NormalMeanVariance(0.3, 0.9),
                q_ω = NormalMeanVariance(0.6, 0.15),
            ),
        )
    end
end
