
@testitem "ImportanceSamplingApproximation: effective sample size is scale-invariant" begin
    using ReactiveMP, BayesBase, Distributions, ExponentialFamily, StableRNGs

    import ReactiveMP: ImportanceSamplingApproximation, approximate_meancov

    # `n_eff` is computed from `bweights`, which at that point holds the raw `g(z)` values --
    # unnormalised, and on whatever scale `g` happens to return. It used to be computed as
    # `1/Σwᵢ²`, the formula that is only valid once the weights sum to one. That made the
    # `n_eff < N/10` resampling threshold depend on the absolute magnitude of `g`: scaling `g`
    # by a constant, which changes nothing about the estimate, flipped resampling on or off.
    #
    # The scale-invariant `(Σwᵢ)²/Σwᵢ²` is used instead. There is no public accessor for
    # `n_eff`, so it is tested through the property that motivated the fix: the returned
    # moments must be invariant under rescaling `g`, since importance weights are only defined
    # up to a constant.

    @testset "rescaling g does not change the estimate" begin
        target = (z) -> exp(-abs2(z - 1.0) / 2)

        # A wide sweep of scales, including ones far outside the range where `1/Σwᵢ²` would
        # land anywhere near `N/10`.
        for scale in (1e-6, 1e-3, 1.0, 1e3, 1e6)
            # A fresh approximation per call, seeded identically, so both runs draw exactly the
            # same samples and any difference is attributable to the weighting alone.
            a = ImportanceSamplingApproximation(StableRNG(42), 2000)
            b = ImportanceSamplingApproximation(StableRNG(42), 2000)

            m_unit, v_unit = approximate_meancov(
                a, target, NormalMeanVariance(0.0, 4.0)
            )
            m_scaled, v_scaled = approximate_meancov(
                b, (z) -> scale * target(z), NormalMeanVariance(0.0, 4.0)
            )

            @test m_unit ≈ m_scaled rtol = 1e-10
            @test v_unit ≈ v_scaled rtol = 1e-10
        end
    end

    @testset "uniform weights give the full sample size, at any scale" begin
        # With a constant `g` every weight is equal, so `n_eff` must equal `N` and no resampling
        # should be triggered -- again independent of the constant's magnitude. Under the old
        # formula `n_eff = 1/(N·c²)`, which for `c = 1` and `N = 1000` gave 1e-3, far below
        # `N/10 = 100`, so resampling fired on perfectly uniform weights.
        for c in (1e-6, 1.0, 1e6)
            approximation = ImportanceSamplingApproximation(StableRNG(7), 1000)
            m, v = approximate_meancov(
                approximation, (z) -> c, NormalMeanVariance(2.0, 3.0)
            )

            # Uniform weights ⇒ the estimate is just the sample mean/variance of the proposal.
            @test m ≈ 2.0 rtol = 0.1
            @test v ≈ 3.0 rtol = 0.15
        end
    end

    @testset "a genuinely degenerate weighting still resamples" begin
        # The fix must not disable resampling: a target concentrated in the far tail of the
        # proposal leaves very few effective samples, and the result should still track the
        # target rather than the proposal.
        approximation = ImportanceSamplingApproximation(StableRNG(11), 5000)
        m, v = approximate_meancov(
            approximation,
            (z) -> exp(-abs2(z - 6.0) / (2 * 0.25)),
            NormalMeanVariance(0.0, 1.0),
        )

        # Pulled well away from the proposal's mean of 0 toward the target's 6.
        @test m > 2.0
        @test isfinite(v)
        @test v > 0
    end
end

@testitem "ImportanceSamplingApproximation: degenerate estimates warn instead of substituting silently" begin
    using ReactiveMP,
        BayesBase, Distributions, ExponentialFamily, StableRNGs, Logging

    import ReactiveMP: ImportanceSamplingApproximation, approximate_meancov

    proposal = NormalMeanVariance(3.0, 7.0)

    @testset "all-zero weights fall back to the proposal, with a warning" begin
        approximation = ImportanceSamplingApproximation(StableRNG(1), 500)

        io = IOBuffer()
        m, v = with_logger(SimpleLogger(io)) do
            approximate_meancov(approximation, (z) -> 0.0, proposal)
        end

        @test m == mean(proposal)
        @test v == var(proposal)
        # The fallback is a confident answer that never saw the target, so it must be visible.
        @test occursin("importance weights are zero", String(take!(io)))
    end

    @testset "the `iszero(v)` guard does not fire on all-identical samples" begin
        # Documents a limitation rather than asserting desired behaviour.
        #
        # The guard includes `iszero(v)`, intended to catch the case where every surviving
        # sample is identical. In practice it almost never fires: `v` is accumulated by
        # `mapreduce` over weighted squared deviations, and floating-point rounding leaves a
        # tiny non-zero residue. With a `PointMass` proposal -- every sample *exactly* equal --
        # the result is v ≈ 1.8e-30, not 0.
        #
        # So a numerically-degenerate variance escapes the guard and can build an effectively
        # singular Gaussian downstream. Choosing a positive threshold instead of `iszero` is a
        # modelling decision (what counts as degenerate?) that belongs to the maintainers, so
        # this test pins the current behaviour rather than changing it. See the discussion on
        # issue #631.
        approximation = ImportanceSamplingApproximation(StableRNG(3), 200)

        m, v = approximate_meancov(approximation, (z) -> 1.0, PointMass(2.0))

        @test m ≈ 2.0
        @test isfinite(v)
        # Not exactly zero -- which is precisely why `iszero(v)` misses it.
        @test !iszero(v)
        @test v < 1e-25
    end

    @testset "an infinite variance is caught, not propagated" begin
        # `isinf(v)` was absent from the guard while `isinf(m)` was present, so an infinite
        # variance escaped into the returned moments. A weight of `Inf` on a single sample
        # produces exactly that. The warning text is asserted in the testset above (shared
        # log site, `maxlog = 1`); here the contract is only that nothing non-finite escapes.
        approximation = ImportanceSamplingApproximation(StableRNG(2), 500)

        m, v = with_logger(SimpleLogger(IOBuffer())) do
            approximate_meancov(
                approximation, (z) -> z > 4.0 ? Inf : 1.0, proposal
            )
        end

        # Whatever happens, a non-finite variance must never be returned.
        @test isfinite(m)
        @test isfinite(v)
        @test v > 0
    end
end

@testitem "ImportanceSamplingApproximation: recovers a known posterior" begin
    using ReactiveMP, BayesBase, Distributions, ExponentialFamily, StableRNGs

    import ReactiveMP: ImportanceSamplingApproximation, approximate_meancov

    # End-to-end accuracy check against an analytically known answer, so the tests above are
    # not the only thing standing between this method and a silent regression.
    #
    # Proposal N(m₀, v₀) reweighted by the Gaussian kernel exp(-(z-μ)²/2σ²) gives, up to
    # normalisation, the Gaussian product N(m₀, v₀)·N(μ, σ²):
    #
    #     v = 1/(1/v₀ + 1/σ²),    m = v·(m₀/v₀ + μ/σ²)
    for (m0, v0, mu, sigma2) in
        ((0.0, 4.0, 1.0, 1.0), (2.0, 1.0, -1.0, 2.0), (1.0, 2.0, 2.0, 1.0))
        v_exact = inv(inv(v0) + inv(sigma2))
        m_exact = v_exact * (m0 / v0 + mu / sigma2)

        approximation = ImportanceSamplingApproximation(
            StableRNG(2024), 200_000
        )
        m, v = approximate_meancov(
            approximation,
            (z) -> exp(-abs2(z - mu) / (2 * sigma2)),
            NormalMeanVariance(m0, v0),
        )

        # Tolerances are Monte Carlo noise floors for 2e5 samples, not slack.
        @test m ≈ m_exact rtol = 2e-2
        @test v ≈ v_exact rtol = 1e-1
    end
end
