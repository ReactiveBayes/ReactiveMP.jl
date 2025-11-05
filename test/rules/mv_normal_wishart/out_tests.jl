
@testitem "rules:MvNormalWishart:out" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules

    @testset "Variational Message Passing: (q_μ::Any, q_λ::Any, q_W::Any, q_ν::Any)" begin
        @test_rules [check_type_promotion = true] MvNormalWishart(:out, Marginalisation) [
            (input = (q_μ = PointMass([1.0]), q_W = PointMass([1.0 0.0;0.0 1.0]), q_λ = PointMass(1.0), q_ν = PointMass(1.0)), output = MvNormalWishart(Float64[1.0], [1.0 0.0;0.0 1.0], 1.0, 1.0)),
            # (input = (q_μ = PointMass([2.0]), q_W = PointMass([2.0]), q_λ = PointMass(1.0), q_ν = PointMass(1.0)), output = MvNormalWishart([2.0], [[2.0 0.0];[0.0 2.0]], 1.0, 1,0)),
            # (input = (q_μ = PointMass([Inf]), q_W = PointMass([0,0]), q_λ = PointMass(1.0), q_ν = PointMass(1.0)), output = MvNormalWishart([Inf], [[0.0 0.0];[0.0 0.0]], 1.0, 1.0)),
        ]
    end
end # testset
