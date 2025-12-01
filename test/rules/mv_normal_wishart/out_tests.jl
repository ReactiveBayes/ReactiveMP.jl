
@testitem "rules:MvNormalWishart:out" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules

    @testset "Variational Message Passing: (q_μ::PointMass, q_λ::PointMass, q_W::PointMass, q_ν::PointMass)" begin
        # Type promotion is false because λ and ν will also be promoted but not necessarily promote μ and W.
        @test_rules [check_type_promotion = false] MvNormalWishart(:out, Marginalisation) [(
            input = (q_μ = PointMass([1.0, 2.0]), q_W = PointMass([1.0 0.0; 0.0 1.0]), q_λ = PointMass(1.0), q_ν = PointMass(1.0)),
            output = MvNormalWishart([1.0, 2.0], [1.0 0.0; 0.0 1.0], 1.0, 1.0)
        ),]
    end
end # testset
