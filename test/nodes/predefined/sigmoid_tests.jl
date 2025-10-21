@testitem "sigmoidNode" begin
    using ReactiveMP, Random, BayesBase, ExponentialFamily
    import ReactiveMP: Sigmoid

    @testset "Average energy" begin
        q_in = NormalMeanVariance(0.0, 1.0)
        for normal_fam in (NormalMeanVariance, NormalMeanPrecision, NormalWeightedMeanPrecision)
            q_in_adj = convert(normal_fam, q_in)
            @test score(
                AverageEnergy(),
                Sigmoid,
                Val{(:out, :in, :ζ)}(),
                (Marginal(Categorical(0.5, 0.5), false, false, nothing), Marginal(q_in_adj, false, false, nothing), Marginal(PointMass(1.0), false, false, nothing)),
                nothing
            ) ≈ 0.8132616875182228
        end
    end
end
