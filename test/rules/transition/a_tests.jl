
@testitem "rules:Transition:a" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules

    @testset "Variational Bayes: (q_out::Any, q_in::Categorical)" begin
        @test_rules [check_type_promotion = false] Transition(:a, Marginalisation) [
            (input = (q_out = Categorical([0.2, 0.5, 0.3]), q_in = Categorical([0.1, 0.4, 0.5])), output = MatrixDirichlet([1.02 1.08 1.1; 1.05 1.2 1.25; 1.03 1.12 1.15])),
            (input = (q_out = PointMass([0.2, 0.5, 0.3]), q_in = Categorical([0.1, 0.4, 0.5])), output = MatrixDirichlet([1.02 1.08 1.1; 1.05 1.2 1.25; 1.03 1.12 1.15])),
            (input = (q_out = Bernoulli(0.3), q_in = Categorical([0.4, 0.6])), output = MatrixDirichlet([1.28 1.42; 1.12 1.18]))
        ]
    end

    @testset "Variational Bayes: (q_out_in::Contingency)" begin
        @test_rules [check_type_promotion = false] Transition(:a, Marginalisation) [(
            input = (q_out_in = Contingency(diageye(3)),), output = MatrixDirichlet([1.333333333333333 1 1; 1 1.3333333333333 1; 1 1 1.33333333333333333])
        )]
    end

    @testset "Variational Bayes: (q_out_in_t1::Contingency)" begin
        @test_rules [check_type_promotion = false] Transition(:a, Marginalisation) [(
            input = (q_out_in_t1 = Contingency(ones(3, 3, 3)),), output = TensorDirichlet(ones(3, 3, 3) .+ (1 / 27))
        )]
    end

    @testset "Variational Bayes: (q_out_in_t1_t2::Contingency)" begin
        @test_rules [check_type_promotion = false] Transition(:a, Marginalisation) [(
            input = (q_out_in_t1_t2 = Contingency(ones(3, 3, 3, 3)),), output = TensorDirichlet(ones(3, 3, 3, 3) .+ (1 / 81))
        )]
    end
end
