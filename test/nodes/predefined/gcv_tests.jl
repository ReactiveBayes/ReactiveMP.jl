
@testitem "ExponentialLinearQuadratic" begin
    using ReactiveMP,
        Distributions, Random, DomainIntegrals, BayesBase, ExponentialFamily

    import ReactiveMP: ExponentialLinearQuadratic

    @testset "Statistics" begin
        approximation = GaussHermiteCubature(11)
        rng = MersenneTwister(123)

        for _ in 1:3
            a, b, c, d = rand(rng, 4)
            dist = ExponentialLinearQuadratic(approximation, a, b, c, d)

            μ, v = ReactiveMP.approximate_meancov(
                approximation,
                (x) ->
                    exp(-(a * x + b * exp(c * x + d * x^2 / 2)) / 2) *
                    exp(x^2 / 2),
                NormalMeanVariance(0.0, 1.0),
            )

            σ = sqrt(v)
            w = inv(v)
            ξ = w * μ

            @test mean(dist) ≈ μ
            @test var(dist) ≈ v
            @test cov(dist) ≈ v
            @test std(dist) ≈ σ
            @test weightedmean(dist) ≈ ξ
            @test invcov(dist) ≈ w
            @test precision(dist) ≈ w

            @test all(mean_var(dist) .≈ (μ, v))
            @test all(mean_cov(dist) .≈ (μ, v))
            @test all(mean_invcov(dist) .≈ (μ, w))
            @test all(mean_precision(dist) .≈ (μ, w))
            @test all(mean_std(dist) .≈ (μ, σ))

            @test all(weightedmean_var(dist) .≈ (ξ, v))
            @test all(weightedmean_cov(dist) .≈ (ξ, v))
            @test all(weightedmean_invcov(dist) .≈ (ξ, w))
            @test all(weightedmean_precision(dist) .≈ (ξ, w))
            @test all(weightedmean_std(dist) .≈ (ξ, σ))
        end
    end

    @testset "prod" begin
        approximation = GaussHermiteCubature(51)
        rng = MersenneTwister(1234)

        for _ in 1:3
            a, b, c, d = rand(rng, 4)
            left = ExponentialLinearQuadratic(approximation, a, b, c, d)

            μ, v = 5.0 * randn(rng), 5.0 * rand(rng)
            right = NormalMeanVariance(μ, v)

            q = NormalMeanVariance(
                ReactiveMP.approximate_meancov(
                    approximation,
                    (x) -> exp(logpdf(left, x) + logpdf(right, x) + x^2 / 2),
                    NormalMeanVariance(0.0, 1.0),
                )...,
            )

            @test all(
                isapprox.(
                    mean_var(q),
                    mean_var(prod(GenericProd(), left, right)),
                    atol = 1e-2,
                ),
            )
            @test all(
                isapprox.(
                    mean_var(q),
                    mean_var(prod(GenericProd(), right, left)),
                    atol = 1e-2,
                ),
            )
        end
    end
end

@testitem "GCV: average energy" begin
    using ReactiveMP, BayesBase, Distributions, ExponentialFamily, StableRNGs

    import ReactiveMP: GCVMetadata, GaussHermiteCubature
    import StatsFuns: log2π

    # The GCV node is `p(y | x, z, κ, ω) = N(y | x, exp(κz + ω))`, so
    #
    #     -log p = ½[log2π + (κz + ω) + (y - x)²·e^{-(κz + ω)}]
    #
    # and the average energy ⟨-log p⟩ under a factorized q is
    #
    #     ½[log2π + ⟨κ⟩⟨z⟩ + ⟨ω⟩ + ψ·A·B]
    #
    # with ψ = ⟨(y - x)²⟩, A = ⟨e^{-ω}⟩ and B the node's Gaussian-moment approximation to
    # ⟨e^{-κz}⟩ (see test/rules/gcv/gcv_setup_tests.jl for why B is approximate). Written out
    # here independently of the implementation.
    function reference_ae(psi, q_z, q_κ, q_ω)
        m_z, v_z = mean_var(q_z)
        m_κ, v_κ = mean_var(q_κ)
        m_ω, v_ω = mean_var(q_ω)
        ksi = m_κ^2 * v_z + m_z^2 * v_κ + v_κ * v_z
        A = exp(-m_ω + v_ω / 2)
        B = exp(-m_κ * m_z + ksi / 2)
        return (log2π + (m_z * m_κ + m_ω) + psi * A * B) / 2
    end

    meanfield_ae(q_y, q_x, q_z, q_κ, q_ω) = score(
        AverageEnergy(),
        GCV,
        Val{(:y, :x, :z, :κ, :ω)}(),
        map(q -> Marginal(q, false, false), (q_y, q_x, q_z, q_κ, q_ω)),
        GCVMetadata(GaussHermiteCubature(20)),
    )

    structured_ae(q_y_x, q_z, q_κ, q_ω) = score(
        AverageEnergy(),
        GCV,
        Val{(:y_x, :z, :κ, :ω)}(),
        map(q -> Marginal(q, false, false), (q_y_x, q_z, q_κ, q_ω)),
        GCVMetadata(GaussHermiteCubature(20)),
    )

    parameter_sets = (
        (
            NormalMeanVariance(3.0, 1.0),
            NormalMeanVariance(1.0, 2.0),
            NormalMeanVariance(0.5, 0.7),
            NormalMeanVariance(0.8, 0.4),
            NormalMeanVariance(1.2, 0.5),
        ),
        (
            NormalMeanVariance(0.4, 0.6),
            NormalMeanVariance(0.5, 0.3),
            NormalMeanVariance(2.0, 0.3),
            NormalMeanVariance(1.2, 0.25),
            NormalMeanVariance(-0.5, 0.8),
        ),
        (
            NormalMeanVariance(-1.5, 0.25),
            NormalMeanVariance(2.5, 0.5),
            NormalMeanVariance(-0.75, 1.25),
            NormalMeanVariance(0.3, 0.9),
            NormalMeanVariance(0.6, 0.15),
        ),
    )

    @testset "Mean-field variant against the derived reference" begin
        for (q_y, q_x, q_z, q_κ, q_ω) in parameter_sets
            psi = (mean(q_y) - mean(q_x))^2 + var(q_y) + var(q_x)
            @test meanfield_ae(q_y, q_x, q_z, q_κ, q_ω) ≈
                reference_ae(psi, q_z, q_κ, q_ω)
        end
    end

    @testset "Structured variant against the derived reference" begin
        for (q_y, q_x, q_z, q_κ, q_ω) in parameter_sets
            V = [var(q_y) 0.3; 0.3 var(q_x)]
            q_y_x = MvNormalMeanCovariance([mean(q_y), mean(q_x)], V)
            psi =
                (mean(q_y) - mean(q_x))^2 + V[1, 1] + V[2, 2] - V[1, 2] -
                V[2, 1]
            @test structured_ae(q_y_x, q_z, q_κ, q_ω) ≈
                reference_ae(psi, q_z, q_κ, q_ω)
        end
    end

    @testset "The two variants agree at zero y-x covariance" begin
        # With Cov(y, x) = 0 the structured q(y, x) carries exactly the information of the
        # factorized pair, so the two @average_energy methods must agree. This is the same
        # invariant that caught the softdot mean-field bug in #615.
        for (q_y, q_x, q_z, q_κ, q_ω) in parameter_sets
            q_y_x = MvNormalMeanCovariance(
                [mean(q_y), mean(q_x)], [var(q_y) 0.0; 0.0 var(q_x)]
            )
            @test meanfield_ae(q_y, q_x, q_z, q_κ, q_ω) ≈
                structured_ae(q_y_x, q_z, q_κ, q_ω)
        end
    end

    @testset "Reduces to the Gaussian entropy term for a deterministic unit variance" begin
        # If κz + ω is pinned to 0, the likelihood collapses to N(y | x, 1) and the average
        # energy must reduce to ½[log2π + ⟨(y-x)²⟩] -- the plain Gaussian cross-entropy.
        q_y = NormalMeanVariance(3.0, 1.0)
        q_x = NormalMeanVariance(1.0, 2.0)
        # `q_z` must be a `NormalDistributionsFamily` per the rule signature; a zero-variance
        # Normal is the degenerate member of that family. `q_κ`/`q_ω` are typed `Any`.
        q_z = NormalMeanVariance(0.0, 0.0)
        q_κ = PointMass(0.0)
        q_ω = PointMass(0.0)

        psi = (3.0 - 1.0)^2 + 1.0 + 2.0
        @test meanfield_ae(q_y, q_x, q_z, q_κ, q_ω) ≈ (log2π + psi) / 2
    end

    @testset "Monte Carlo confirmation of the exactly-computable part" begin
        # With κ = z = 0 the only stochastic term left is ω, whose contribution
        # ⟨ω⟩ + ψ·⟨e^{-ω}⟩ is exact (no Gaussian-moment approximation is involved). Confirms
        # the average energy against sampling rather than against a rearranged formula.
        rng = StableRNG(42)
        q_y = NormalMeanVariance(3.0, 1.0)
        q_x = NormalMeanVariance(1.0, 2.0)
        q_z = NormalMeanVariance(0.0, 0.0)
        q_κ = PointMass(0.0)
        q_ω = NormalMeanVariance(0.7, 0.4)

        psi = (3.0 - 1.0)^2 + 1.0 + 2.0
        ω_samples = rand(rng, Normal(0.7, sqrt(0.4)), 10^6)
        mc = mean(@. (log2π + ω_samples + psi * exp(-ω_samples)) / 2)

        @test meanfield_ae(q_y, q_x, q_z, q_κ, q_ω) ≈ mc rtol = 5e-3
    end
end
