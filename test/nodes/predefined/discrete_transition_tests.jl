@testitem "DiscreteTransitionNode" begin
    using Test, ReactiveMP, Random, Distributions, BayesBase, ExponentialFamily

    @testset "DiscreteTransition node properties" begin
        @test ReactiveMP.sdtype(DiscreteTransition) == Stochastic()
        @test ReactiveMP.alias_interface(DiscreteTransition, 1, :out) == :out
        @test ReactiveMP.alias_interface(DiscreteTransition, 2, :in) == :in
        @test ReactiveMP.alias_interface(DiscreteTransition, 3, :in) == :a
        @test ReactiveMP.alias_interface(DiscreteTransition, 4, :in) == :T1

        @test ReactiveMP.collect_factorisation(DiscreteTransition, ()) == ()
    end
    @testset "AverageEnergy(q_out_in::Contingency, q_a::DirichletCollection)" begin end

    @testset "AverageEnergy(q_out_in::Contingency, q_a::PointMass)" begin
        contingency_matrix = [0.2 0.3; 0.4 0.1]
        a_matrix = [0.7 0.3; 0.2 0.8]

        q_out_in = Contingency(contingency_matrix)
        q_a = PointMass(a_matrix)

        marginals = (Marginal(q_out_in, false, false, nothing), Marginal(q_a, false, false, nothing))

        # Expected value calculated by hand
        expected = -sum(contingency_matrix .* log.(clamp.(a_matrix, tiny, Inf)))

        @test score(AverageEnergy(), DiscreteTransition, Val{(:out_in, :a)}(), marginals, nothing) ≈ expected

        contingency_matrix = [0.2 0.3; 0.4 0.1]
        a_matrix = [1.0 0.0; 0.0 1.0]

        q_out_in = Contingency(contingency_matrix)
        q_a = PointMass(a_matrix)

        marginals = (Marginal(q_out_in, false, false, nothing), Marginal(q_a, false, false, nothing))

        expected = -sum(contingency_matrix .* log.(clamp.(a_matrix, tiny, Inf)))

        @test score(AverageEnergy(), DiscreteTransition, Val{(:out_in, :a)}(), marginals, nothing) ≈ expected

        contingency_matrix = prod.(Iterators.product([0, 1, 0], [0.1, 0.4, 0.5]))
        a_matrix = diageye(3)

        q_out_in = Contingency(contingency_matrix)
        q_a = PointMass(a_matrix)

        marginals = (Marginal(q_out_in, false, false, nothing), Marginal(q_a, false, false, nothing))

        expected = -sum(contingency_matrix .* log.(clamp.(a_matrix, tiny, Inf)))

        @test score(AverageEnergy(), DiscreteTransition, Val{(:out_in, :a)}(), marginals, nothing) ≈ expected

        contingency_matrix = [0.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 0.0]
        q_out_in = Contingency(contingency_matrix)
        q_a = PointMass(diageye(3))

        marginals = (Marginal(q_out_in, false, false, nothing), Marginal(q_a, false, false, nothing))

        expected = -sum(contingency_matrix .* log.(clamp.(a_matrix, tiny, Inf)))
        @test score(AverageEnergy(), DiscreteTransition, Val{(:out_in, :a)}(), marginals, nothing) ≈ expected
    end

    @testset "AverageEnergy(q_out::Any, q_in::Any, q_a::PointMass)" begin
        q_out = Categorical([0.3, 0.7])
        q_in = Categorical([0.8, 0.2])
        q_a = PointMass([0.7 0.3; 0.2 0.8])

        marginals = (Marginal(q_out, false, false, nothing), Marginal(q_in, false, false, nothing), Marginal(q_a, false, false, nothing))

        contingency = probvec(q_out) * probvec(q_in)'
        expected = -sum(contingency .* log.(clamp.(mean(q_a), tiny, Inf)))

        @test score(AverageEnergy(), DiscreteTransition, Val{(:out, :in, :a)}(), marginals, nothing) ≈ expected

        q_out = Categorical([0.0, 1.0])
        q_in = Categorical([0.0, 1.0])
        q_a = PointMass([1.0 0.0; 1.0 0.0])

        marginals = (Marginal(q_out, false, false, nothing), Marginal(q_in, false, false, nothing), Marginal(q_a, false, false, nothing))

        contingency = probvec(q_out) * probvec(q_in)'

        expected = -sum(contingency .* log.(clamp.(mean(q_a), tiny, Inf)))
        @test score(AverageEnergy(), DiscreteTransition, Val{(:out, :in, :a)}(), marginals, nothing) ≈ expected
    end
end
