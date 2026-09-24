# From v6's `test/nodes/predefined/autoregressive_tests.jl`: the node's average energy, its
# algorithm and its noise matrices.

@testitem "AR: average energy" tags = [:rules] begin
    using AutoregressiveMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, LinearAlgebra, StableRNGs

    diageye(n) = Matrix{Float64}(I, n, n)

    # v6 compared its energies without type promotion; `0.5log2π` is a Float64 literal.
    @test_average_energy(
        node = AR, algorithm = ARVMP(Univariate, 1, ARsafe()), check_type_promotion = false,
        cases = [
            (q = (y = NormalMeanVariance(0.0, 1.0), x = NormalMeanVariance(0.0, 1.0), θ = NormalMeanVariance(0.0, 1.0), γ = GammaShapeRate(2.0, 3.0)),) => 1.92351917665,
            (q = (y = MvNormalMeanCovariance(zeros(2), diageye(2)), x = MvNormalMeanCovariance(zeros(2), diageye(2)), θ = MvNormalMeanCovariance(zeros(2), diageye(2)), γ = GammaShapeRate(2.0, 3.0)),) => 2.25685250999,
            (clusters = ((:y, :x) => MvNormalMeanCovariance(zeros(2), diageye(2)),), q = (θ = NormalMeanVariance(0.0, 1.0), γ = GammaShapeRate(2.0, 3.0))) => 1.92351917665616,
        ],
    )

    # v6 left the multivariate energies untested. With q(y, x) the product of q(y) and q(x),
    # the structured energy is the mean-field one: the cross-covariance is zero, and both
    # corrections take out the entropy of y[2:end].
    @testset "multivariate: structured over independent q(y) q(x) is mean-field" begin
        rng = StableRNG(123)
        for order in (2, 3)
            algorithm = ARVMP(Multivariate, order, ARsafe())
            A, B, C = randn(rng, order, order), randn(rng, order, order), randn(rng, order, order)
            q_y = MvNormalMeanCovariance(randn(rng, order), A * A' + I)
            q_x = MvNormalMeanCovariance(randn(rng, order), B * B' + I)
            q_θ = MvNormalMeanCovariance(randn(rng, order), C * C' + I)
            q_γ = GammaShapeRate(2.0, 3.0)
            V = zeros(2order, 2order)
            V[1:order, 1:order] = cov(q_y)
            V[(order + 1):end, (order + 1):end] = cov(q_x)
            q_y_x = MvNormalMeanCovariance([mean(q_y); mean(q_x)], V)

            meanfield = call_average_energy(AR; q = (y = q_y, x = q_x, θ = q_θ, γ = q_γ), algorithm)
            structured = call_average_energy(AR; clusters = ((:y, :x) => q_y_x,), q = (θ = q_θ, γ = q_γ), algorithm)
            @test isfinite(meanfield)
            @test structured ≈ meanfield
        end
    end
end

@testitem "AR: ARVMP" tags = [:rules] begin
    using AutoregressiveMessagePassingRules, MessagePassingRulesBase, Distributions, Test
    using AutoregressiveMessagePassingRules: getvform, getorder, getstype, is_univariate, is_multivariate, is_safe, is_unsafe

    algo_uni = ARVMP(Univariate, 1, ARsafe())
    algo_multi = ARVMP(Multivariate, 3, ARunsafe())

    @test algo_uni isa AbstractAlgorithm
    @test getvform(algo_uni) === Univariate
    @test getorder(algo_uni) == 1
    @test getstype(algo_uni) === ARsafe()
    @test is_univariate(algo_uni)
    @test !is_multivariate(algo_uni)
    @test is_safe(algo_uni)
    @test !is_unsafe(algo_uni)

    @test getvform(algo_multi) === Multivariate
    @test getorder(algo_multi) == 3
    @test is_multivariate(algo_multi)
    @test !is_univariate(algo_multi)
    @test is_unsafe(algo_multi)
    @test !is_safe(algo_multi)

    # A univariate AR is an AR(1), with a warning for any other order.
    algo = @test_logs (:warn, r"order is forced to 1") ARVMP(Univariate, 3, ARsafe())
    @test getorder(algo) == 1

    # v6's `default_meta(AR)` threw; the node declares no algorithm, and its default has no rules.
    @test MessagePassingRulesBase.default_algorithm(AR) === DefaultAlgorithm()
    @test Autoregressive === AR
    @test MessagePassingRulesBase.alias_interface(AR, :out) === :y
    @test MessagePassingRulesBase.alias_interface(ConjugateAR, :out) === :y
end

@testitem "AR: noise matrices and ar_unit" tags = [:rules] begin
    using AutoregressiveMessagePassingRules, BayesBase, Distributions, LinearAlgebra, StableRNGs
    using AutoregressiveMessagePassingRules: ARTransitionMatrix, ARPrecisionMatrix, ar_transition, ar_precision, add_transition, add_transition!, add_precision, add_precision!, ar_unit, StandardBasisVector
    using BayesBase: huge

    @testset "ARTransitionMatrix" begin
        rng = StableRNG(1233)
        for γ in randn(rng, 3), order in 2:4
            transition = ARTransitionMatrix(order, γ)
            matrix = rand(rng, order, order)
            ftransition = zeros(order, order)
            ftransition[1] = inv(γ)

            @test transition == ftransition
            @test broadcast(+, matrix, transition) == (matrix + ftransition)
            @test_throws DimensionMismatch broadcast(+, zeros(order + 1, order + 1), transition)

            @test add_transition(matrix, transition) == (matrix + ftransition)
            @test_throws DimensionMismatch add_transition(zeros(order + 1, order + 1), transition)

            # v6 overloaded `broadcast!(+, matrix, transition)` to add in place; the port keeps
            # broadcasting's own meaning and adds in place with `add_transition!`.
            cmatrix = copy(matrix)
            @test add_transition!(cmatrix, transition) === cmatrix
            @test cmatrix == (matrix + ftransition)
            @test_throws DimensionMismatch add_transition!(zeros(order + 1, order + 1), transition)

            @test convert(AbstractArray{Float32}, transition) isa ARTransitionMatrix{Float32}
        end
        @test ar_transition(Univariate, 1, 2.0) == 0.5
        @test ar_transition(Multivariate, 2, 2.0) == [0.5 0.0; 0.0 0.0]
        @test add_transition(1.0, 2.0) == 3.0
        @test add_transition!(1.0, 2.0) == 3.0
        # From an integer precision, a float inverse.
        @test ARTransitionMatrix(2, 2) == [0.5 0.0; 0.0 0.0]
    end

    @testset "ARPrecisionMatrix" begin
        γ = 2.5
        order = 3
        pm = ar_precision(Multivariate, order, γ)
        @test pm isa ARPrecisionMatrix
        @test size(pm) == (3, 3)
        @test pm[1, 1] == γ
        @test pm[2, 2] ≈ convert(Float64, huge)
        @test pm[1, 2] == 0.0
        @test ar_precision(Univariate, 1, γ) == γ

        A = zeros(3, 3)
        res = add_precision(A, pm)
        @test res[1, 1] == γ
        @test res[2, 2] ≈ huge
        @test A == zeros(3, 3)

        B = zeros(3, 3)
        @test add_precision!(B, pm) === B
        @test B[1, 1] == γ
        @test all(diag(B)[2:end] .≈ huge)
        @test_throws DimensionMismatch add_precision!(zeros(2, 2), pm)

        @test add_precision!(1.0, 2.0) == 3.0
        @test add_precision(1.0, 2.0) == 3.0
        @test convert(AbstractArray{Float32}, pm) isa ARPrecisionMatrix{Float32}
    end

    @testset "ar_unit" begin
        @test ar_unit(Univariate, 1) == 1.0
        @test ar_unit(Multivariate, 3) isa StandardBasisVector{Float64}
        @test ar_unit(Multivariate, 3) == [1.0, 0.0, 0.0]
        @test ar_unit(Float32, Univariate, 1) === 1.0f0
        @test eltype(ar_unit(Float32, Multivariate, 2)) === Float32
    end
end
