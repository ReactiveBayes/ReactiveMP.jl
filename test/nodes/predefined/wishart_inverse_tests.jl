
@testitem "InverseWishartNode" begin
    using ReactiveMP, Random, BayesBase, ExponentialFamily, Distributions

    @testset "AverageEnergy" begin
        begin
            q_out = InverseWishart(2.0, [2.0 0.0; 0.0 2.0])
            q_ν  = PointMass(2.0)
            q_S   = PointMass([2.0 0.0; 0.0 2.0])

            marginals = (Marginal(q_out, false, false, nothing), Marginal(q_ν, false, false, nothing), Marginal(q_S, false, false, nothing))
            @test score(AverageEnergy(), InverseWishart, Val{(:out, :ν, :S)}(), marginals, nothing) ≈ 9.496544113156787
        end

        begin
            S     = [4.3082195553088445 0.4573472347695425 -2.748089173206861; 0.4573472347695425 0.0954087613417567 -0.29586598556052124; -2.748089173206861 -0.29586598556052124 2.9875706318257538]
            ν    = 4.0
            q_out = InverseWishart(ν, S)
            q_ν  = PointMass(ν)
            q_S   = PointMass(S)

            marginals = (Marginal(q_out, false, false, nothing), Marginal(q_ν, false, false, nothing), Marginal(q_S, false, false, nothing))
            @test score(AverageEnergy(), InverseWishart, Val{(:out, :ν, :S)}(), marginals, nothing) ≈ 1.1299587008097587
        end
    end
end
