# v6's `test/rules/multinomial_polya/{psi_tests,x_tests}.jl`, under `MultinomialPolyaApproximation()`,
# v6's `MultinomialPolyaMeta(21)`, and with v6's `check_type_promotion = false`. The rules take
# the mode of q(N), and each q(N) below has mode 3. v6's node test checked the average energy
# only, and `energy_tests.jl` replaces it.

@testitem "rules:MultinomialPolya:ψ" tags = [:rules] begin
    using PolyaMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using LinearAlgebra: I, Diagonal

    # η = x[1:K-1] - N_k/2 and Λ = Diagonal(ω), ω the Pólya-Gamma mean at ⟨ψ⟩ = 0, N_k/4.
    q_x = PointMass([0, 1, 2])
    m = (ψ = MvNormalWeightedMeanPrecision(zeros(2), Matrix(1.0I, 2, 2)),)
    expected = MvNormalWeightedMeanPrecision([-1.5, -0.5], Diagonal([0.75, 0.75]))

    @testset "(q_x::PointMass, q_N::$(typeof(q_N)), m_ψ::MvNormalWeightedMeanPrecision)" for q_N in (PointMass(3), Poisson(3), Binomial(5, 0.5), Categorical([0.01, 0.01, 0.98]))
        @test_message_update_rule(
            node = MultinomialPolya, target = :ψ, algorithm = MultinomialPolyaApproximation(), check_type_promotion = false,
            cases = [(q = (x = q_x, N = q_N), m = m) => expected],
        )
    end
end

@testitem "rules:MultinomialPolya:x" tags = [:rules] begin
    using PolyaMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions

    # Logistic stick-breaking at ⟨ψ⟩ = 0: 1/2, then 1/2 of the rest, then the rest.
    q_ψ = MvNormalWeightedMeanPrecision([0.0, 0.0], [1.0 0.0; 0.0 1.0])
    expected = Multinomial(3, [0.5, 0.25, 0.25])

    @testset "(q_N::$(typeof(q_N)), q_ψ::MvNormalWeightedMeanPrecision)" for q_N in (PointMass(3), Poisson(3), Binomial(5, 0.5), Categorical([0.01, 0.01, 0.98]))
        @test_message_update_rule(
            node = MultinomialPolya, target = :x, algorithm = MultinomialPolyaApproximation(), check_type_promotion = false,
            cases = [(q = (N = q_N, ψ = q_ψ),) => expected],
        )
    end
end
