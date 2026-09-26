# The node's distribution ExponentialLinearQuadratic and its average energy.

@testitem "ExponentialLinearQuadratic" tags = [:rules] begin
    using GCVMessagePassingRules, BayesBase, ExponentialFamily, Distributions, StableRNGs
    using MessagePassingRulesApproximations: GaussHermiteCubature, approximate_meancov

    @testset "Statistics" begin
        approximation = GaussHermiteCubature(11)
        rng = StableRNG(123)
        for _ in 1:3
            a, b, c, d = rand(rng, 4)
            dist = ExponentialLinearQuadratic(approximation, a, b, c, d)
            μ, v = approximate_meancov(approximation, (x) -> exp(-(a * x + b * exp(c * x + d * x^2 / 2)) / 2) * exp(x^2 / 2), 0.0, 1.0)
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
        rng = StableRNG(1234)
        for _ in 1:3
            a, b, c, d = rand(rng, 4)
            left = ExponentialLinearQuadratic(approximation, a, b, c, d)
            μ, v = 5.0 * randn(rng), 5.0 * rand(rng)
            right = NormalMeanVariance(μ, v)
            q = NormalMeanVariance(approximate_meancov(approximation, (x) -> exp(logpdf(left, x) + logpdf(right, x) + x^2 / 2), 0.0, 1.0)...)
            # The product's cubature is centred on the normal, where the doubly exponential
            # factor is poorly resolved: for these draws it is off by up to 3.5e-2 in the
            # variance (the reference agrees with a fine grid to 1e-3), so a tolerance of 1e-2
            # would hold only for particular draws.
            @test all(isapprox.(mean_var(q), mean_var(prod(GenericProd(), left, right)), atol = 5.0e-2))
            @test all(isapprox.(mean_var(q), mean_var(prod(GenericProd(), right, left)), atol = 5.0e-2))
        end
    end
end

@testitem "GCV: average energy" tags = [:rules] begin
    using GCVMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, BayesBase, Distributions, ExponentialFamily, StableRNGs
    using MessagePassingRulesApproximations: GaussHermiteCubature
    using StatsFuns: log2π

    # The GCV node is `p(y | x, z, κ, ω) = N(y | x, exp(κz + ω))`, so
    #
    #     -log p = ½[log2π + (κz + ω) + (y - x)²·e^{-(κz + ω)}]
    #
    # and the average energy ⟨-log p⟩ under a factorized q is
    #
    #     ½[log2π + ⟨κ⟩⟨z⟩ + ⟨ω⟩ + ψ·A·B]
    #
    # with ψ = ⟨(y - x)²⟩, A = ⟨e^{-ω}⟩ and B the node's Gaussian-moment approximation to
    # ⟨e^{-κz}⟩ (see `gcv_setup.jl` for why B is approximate). Written out here independently
    # of the implementation.
    function reference_ae(psi, q_z, q_κ, q_ω)
        m_z, v_z = mean_var(q_z)
        m_κ, v_κ = mean_var(q_κ)
        m_ω, v_ω = mean_var(q_ω)
        ksi = m_κ^2 * v_z + m_z^2 * v_κ + v_κ * v_z
        A = exp(-m_ω + v_ω / 2)
        B = exp(-m_κ * m_z + ksi / 2)
        return (log2π + (m_z * m_κ + m_ω) + psi * A * B) / 2
    end

    algorithm = GCVApproximation(method = GaussHermiteCubature(20))
    meanfield_ae(q_y, q_x, q_z, q_κ, q_ω) = getresult(call_average_energy(GCV; q = (y = q_y, x = q_x, z = q_z, κ = q_κ, ω = q_ω), algorithm))
    structured_ae(q_y_x, q_z, q_κ, q_ω) = getresult(call_average_energy(GCV; clusters = ((:y, :x) => q_y_x,), q = (z = q_z, κ = q_κ, ω = q_ω), algorithm))

    parameter_sets = (
        (NormalMeanVariance(3.0, 1.0), NormalMeanVariance(1.0, 2.0), NormalMeanVariance(0.5, 0.7), NormalMeanVariance(0.8, 0.4), NormalMeanVariance(1.2, 0.5)),
        (NormalMeanVariance(0.4, 0.6), NormalMeanVariance(0.5, 0.3), NormalMeanVariance(2.0, 0.3), NormalMeanVariance(1.2, 0.25), NormalMeanVariance(-0.5, 0.8)),
        (NormalMeanVariance(-1.5, 0.25), NormalMeanVariance(2.5, 0.5), NormalMeanVariance(-0.75, 1.25), NormalMeanVariance(0.3, 0.9), NormalMeanVariance(0.6, 0.15)),
    )

    @testset "Mean-field variant against the derived reference" begin
        for (q_y, q_x, q_z, q_κ, q_ω) in parameter_sets
            psi = (mean(q_y) - mean(q_x))^2 + var(q_y) + var(q_x)
            @test meanfield_ae(q_y, q_x, q_z, q_κ, q_ω) ≈ reference_ae(psi, q_z, q_κ, q_ω)
        end
        # And as a table, with Float32 and BigFloat inputs.
        (q_y, q_x, q_z, q_κ, q_ω) = first(parameter_sets)
        @test_average_energy(
            node = GCV, algorithm = algorithm,
            cases = [(q = (y = q_y, x = q_x, z = q_z, κ = q_κ, ω = q_ω),) => reference_ae(7.0, q_z, q_κ, q_ω)],
        )
    end

    @testset "Structured variant against the derived reference" begin
        for (q_y, q_x, q_z, q_κ, q_ω) in parameter_sets
            V = [var(q_y) 0.3; 0.3 var(q_x)]
            q_y_x = MvNormalMeanCovariance([mean(q_y), mean(q_x)], V)
            psi = (mean(q_y) - mean(q_x))^2 + V[1, 1] + V[2, 2] - V[1, 2] - V[2, 1]
            @test structured_ae(q_y_x, q_z, q_κ, q_ω) ≈ reference_ae(psi, q_z, q_κ, q_ω)
        end
    end

    @testset "The two variants agree at zero y-x covariance" begin
        # With Cov(y, x) = 0 the structured q(y, x) carries exactly the information of the
        # factorized pair, so the two average energies must agree.
        for (q_y, q_x, q_z, q_κ, q_ω) in parameter_sets
            q_y_x = MvNormalMeanCovariance([mean(q_y), mean(q_x)], [var(q_y) 0.0; 0.0 var(q_x)])
            @test meanfield_ae(q_y, q_x, q_z, q_κ, q_ω) ≈ structured_ae(q_y_x, q_z, q_κ, q_ω)
        end
    end

    @testset "Reduces to the Gaussian entropy term for a deterministic unit variance" begin
        # If κz + ω is pinned to 0, the likelihood collapses to N(y | x, 1) and the average
        # energy must reduce to ½[log2π + ⟨(y-x)²⟩] -- the plain Gaussian cross-entropy.
        # `q_z` must be a `NormalDistributionsFamily` per the energy's signature; a
        # zero-variance Normal is the degenerate member of that family. `q_κ`/`q_ω` are `Any`.
        psi = (3.0 - 1.0)^2 + 1.0 + 2.0
        @test meanfield_ae(NormalMeanVariance(3.0, 1.0), NormalMeanVariance(1.0, 2.0), NormalMeanVariance(0.0, 0.0), PointMass(0.0), PointMass(0.0)) ≈ (log2π + psi) / 2
    end

    @testset "Monte Carlo confirmation of the exactly-computable part" begin
        # With κ = z = 0 the only stochastic term left is ω, whose contribution
        # ⟨ω⟩ + ψ·⟨e^{-ω}⟩ is exact (no Gaussian-moment approximation is involved). Confirms
        # the average energy against sampling rather than against a rearranged formula.
        rng = StableRNG(42)
        psi = (3.0 - 1.0)^2 + 1.0 + 2.0
        ω_samples = rand(rng, Normal(0.7, sqrt(0.4)), 10^6)
        mc = mean(@. (log2π + ω_samples + psi * exp(-ω_samples)) / 2)
        ae = meanfield_ae(NormalMeanVariance(3.0, 1.0), NormalMeanVariance(1.0, 2.0), NormalMeanVariance(0.0, 0.0), PointMass(0.0), NormalMeanVariance(0.7, 0.4))
        @test ae ≈ mc rtol = 5.0e-3
    end
end

@testitem "GCV: the algorithm takes a Gauss–Hermite cubature" tags = [:rules] begin
    using GCVMessagePassingRules, MessagePassingRulesApproximations
    @test GCVApproximation(method = GaussHermiteCubature(8)).method isa GaussHermiteCubature
    # Its rules take moments by cubature, so another method is refused where it is built.
    @test_throws ArgumentError GCVApproximation(method = Unscented())
end
