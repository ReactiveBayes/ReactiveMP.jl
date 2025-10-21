
@testitem "rules:Categorical:p" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules

    @testset "Variational Message Passing: (q_out::PointMass)" begin
        @test_rules [check_type_promotion = true] Categorical(:p, Marginalisation) [(input = (q_out = PointMass([0.0, 1.0]),), output = Dirichlet([1.0, 2.0]))]
    end

    @testset "Variational Message Passing: (q_out::Categorical)" begin
        @test_rules [check_type_promotion = false] Categorical(:p, Marginalisation) [(input = (q_out = Categorical([0.0, 1.0]),), output = Dirichlet([1.0, 2.0]))]
        @test_rules [check_type_promotion = false] Categorical(:p, Marginalisation) [(input = (q_out = Categorical([0.5, 0.5]),), output = Dirichlet([1.5, 1.5]))]
    end

    # Test if the input q_out for the outgoing message towards p edge is not a scalar
    @testset "q_out =! scalar" begin
        @test_throws ArgumentError @call_rule Categorical(:p, Marginalisation) (q_out = PointMass(1.0),)
        @test_throws ArgumentError @call_rule Categorical(:p, Marginalisation) (q_out = PointMass(1),)
        @test_throws ArgumentError @call_rule Categorical(:p, Marginalisation) (q_out = 1.0,)
    end

    # Test if the input q_out for the outgoing message towards p edge is a one-hot encoded vector
    @testset "q_out = PointMass one-hot encoded vector" begin
        # PointMass over a non-one-hot vector
        @test_throws "q_out must be one-hot encoded" @call_rule Categorical(:p, Marginalisation) (q_out = PointMass([0.5, 0.5]),)
        @test_throws "q_out must be one-hot encoded" @call_rule Categorical(:p, Marginalisation) (q_out = PointMass([1.0, 1.0]),)
        # Arbitrary non-distribution
        @test_throws "q_out is only defined for PointMass over a one-hot vector" @call_rule Categorical(:p, Marginalisation) (q_out = "Hello?",)
    end
end
