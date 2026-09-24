# From v6's `test/rules/discrete_transition/a_tests.jl`.
@testitem "rules:DiscreteTransition:a:Variational Bayes: (q_out::Any, q_in::Categorical)" tags = [
    :rules,
] begin
    using DiscreteTransitionMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, LinearAlgebra, Random
    @test_message_update_rule(
        node = DiscreteTransition, target = :a, check_type_promotion = false,
        cases = [
            (q = (out = Categorical([0.2, 0.5, 0.3]), in = Categorical([0.1, 0.4, 0.5])),) => DirichletCollection(
                [1.02 1.08 1.1; 1.05 1.2 1.25; 1.03 1.12 1.15]
            ),
            (q = (out = PointMass([0.2, 0.5, 0.3]), in = Categorical([0.1, 0.4, 0.5])),) => DirichletCollection(
                [1.02 1.08 1.1; 1.05 1.2 1.25; 1.03 1.12 1.15]
            ),
            (q = (out = Bernoulli(0.3), in = Categorical([0.4, 0.6])),) => DirichletCollection([1.28 1.42; 1.12 1.18]),
        ],
    )
end

@testitem "rules:DiscreteTransition:a:Variational Bayes: (q_out_in::Contingency)" tags = [
    :rules,
] begin
    using DiscreteTransitionMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, LinearAlgebra, Random

    diageye(n) = Matrix{Float64}(I, n, n)
    @test_message_update_rule(
        node = DiscreteTransition, target = :a, check_type_promotion = false,
        cases = [
            (clusters = ((:out, :in) => Contingency(diageye(3)),),) => DirichletCollection(
                [
                    1.333333333333333 1 1;
                    1 1.3333333333333 1;
                    1 1 1.33333333333333333
                ],
            ),
        ],
    )
end

@testitem "rules:DiscreteTransition:a:Variational Bayes: (q_out_in_T1::Contingency)" tags = [
    :rules,
] begin
    using DiscreteTransitionMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, LinearAlgebra, Random
    @test_message_update_rule(
        node = DiscreteTransition, target = :a, check_type_promotion = false,
        cases = [
            (clusters = ((:out, :in, :T) => Contingency(ones(3, 3, 3)),),) => DirichletCollection(ones(3, 3, 3) .+ (1 / 27)),
        ],
    )
end

@testitem "rules:DiscreteTransition:a:Variational Bayes: (q_out_in_T1_t2::Contingency)" tags = [
    :rules,
] begin
    using DiscreteTransitionMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, LinearAlgebra, Random
    @test_message_update_rule(
        node = DiscreteTransition, target = :a, check_type_promotion = false,
        cases = [
            (clusters = ((:out, :in, :T) => Contingency(ones(3, 3, 3, 3)),),) => DirichletCollection(ones(3, 3, 3, 3) .+ (1 / 81)),
        ],
    )
end

@testitem "rules:DiscreteTransition:a:Variational Bayes: (q_out_in::Contingency, q_T1::Categorical)" tags = [
    :rules,
] begin
    # This should be the normalized outer product of the marginals, along the decoded dimensions
    using DiscreteTransitionMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, LinearAlgebra, Random
    @test_message_update_rule(
        node = DiscreteTransition, target = :a, check_type_promotion = false,
        cases = [
            (q = (T = (Categorical([0.1, 0.4, 0.5]),),), clusters = ((:out, :in) => Contingency([0.1 0.4 0.5; 0.2 0.5 0.3; 0.3 0.6 0.2]),)) => DirichletCollection(
                [
                    1.0032258064516129 1.0129032258064516 1.0161290322580645; 1.0064516129032257 1.0161290322580645 1.0096774193548388; 1.0096774193548388 1.0193548387096774 1.0064516129032257;;;
                    1.0129032258064516 1.0516129032258064 1.064516129032258; 1.0258064516129033 1.064516129032258 1.038709677419355; 1.038709677419355 1.0774193548387097 1.0258064516129033;;;
                    1.0161290322580645 1.064516129032258 1.0806451612903225; 1.032258064516129 1.0806451612903225 1.0483870967741935; 1.0483870967741935 1.096774193548387 1.032258064516129
                ],
            ),
            (q = (T = (Categorical([0.1, 0.4, 0.5]),),), clusters = ((:out, :in) => Contingency([0.8 0.4 0.5; 0.2 0.5 0.3; 1.0 0.6 0.2]),)) => DirichletCollection(
                [
                    1.017777777777778 1.008888888888889 1.011111111111111; 1.0044444444444445 1.011111111111111 1.0066666666666666; 1.0222222222222221 1.0133333333333334 1.0044444444444445;;;
                    1.0711111111111111 1.0355555555555556 1.0444444444444445; 1.017777777777778 1.0444444444444445 1.0266666666666666; 1.0888888888888888 1.0533333333333332 1.017777777777778;;;
                    1.0888888888888888 1.0444444444444445 1.0555555555555556; 1.0222222222222221 1.0555555555555556 1.0333333333333334; 1.1111111111111112 1.0666666666666667 1.0222222222222221
                ],
            ),
        ],
    )
end

@testitem "rules:DiscreteTransition:a:Variational Bayes: (q_out_in::Contingency, q_T1::PointMass)" tags = [
    :rules,
] begin
    # This should be the normalized outer product of the marginals, along the decoded dimensions
    using DiscreteTransitionMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, LinearAlgebra, Random
    @test_message_update_rule(
        node = DiscreteTransition, target = :a, check_type_promotion = false,
        cases = [
            (q = (T = (PointMass([0.0, 1.0, 0.0]),),), clusters = ((:out, :in) => Contingency([0.1 0.4 0.5; 0.2 0.5 0.3; 0.3 0.6 0.2]),)) => DirichletCollection(
                [
                    1.0 1.0 1.0; 1.0 1.0 1.0; 1.0 1.0 1.0;;;
                    1.032258064516129 1.129032258064516 1.1612903225806452; 1.064516129032258 1.1612903225806452 1.096774193548387; 1.096774193548387 1.1935483870967742 1.064516129032258;;;
                    1.0 1.0 1.0; 1.0 1.0 1.0; 1.0 1.0 1.0
                ],
            ),
            (q = (T = (PointMass([0.0, 1.0, 0.0]),),), clusters = ((:out, :in) => Contingency([0.8 0.4 0.5; 0.2 0.5 0.3; 1.0 0.6 0.2]),)) => DirichletCollection(
                [
                    1.0 1.0 1.0; 1.0 1.0 1.0; 1.0 1.0 1.0;;;
                    1.1777777777777778 1.0888888888888888 1.1111111111111112; 1.0444444444444445 1.1111111111111112 1.0666666666666667; 1.2222222222222223 1.1333333333333333 1.0444444444444445;;;
                    1.0 1.0 1.0; 1.0 1.0 1.0; 1.0 1.0 1.0
                ],
            ),
        ],
    )
end

@testitem "rules:DiscreteTransition:a:Variational Bayes: (q_out_T1::Contingency, q_in::Categorical)" tags = [
    :rules,
] begin
    # This should be the normalized outer product of the marginals, along the decoded dimensions
    using DiscreteTransitionMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, LinearAlgebra, Random
    @test_message_update_rule(
        node = DiscreteTransition, target = :a, check_type_promotion = false,
        cases = [
            (q = (in = Categorical([0.1, 0.4, 0.5]),), clusters = ((:out, :T) => Contingency([0.8 0.4 0.5; 0.2 0.5 0.3; 1.0 0.6 0.2]),)) => DirichletCollection(
                [
                    1.017777777777778 1.0711111111111111 1.0888888888888888; 1.0044444444444445 1.017777777777778 1.0222222222222221; 1.0222222222222221 1.0888888888888888 1.1111111111111112;;;
                    1.008888888888889 1.0355555555555556 1.0444444444444445; 1.011111111111111 1.0444444444444445 1.0555555555555556; 1.0133333333333334 1.0533333333333332 1.0666666666666667;;;
                    1.011111111111111 1.0444444444444445 1.0555555555555556; 1.0066666666666666 1.0266666666666666 1.0333333333333334; 1.0044444444444445 1.017777777777778 1.0222222222222221
                ],
            ),
        ],
    )
end
