@testitem "rules:Sigmoid:out" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions
    using StatsFuns: logistic

    import ReactiveMP: @test_rules

    @testset "Mean Field: (q_in::UnivariateNormalDistributionsFamily, q_ζ::PointMass)" begin
        q_in = [NormalMeanVariance(0.0, 1.0), NormalMeanVariance(-1.0, 1.0), NormalMeanVariance(10.0, 1.0)]
        results = [[0.5, 0.5], [0.2689414213699951, 0.7310585786300049], [0.9999546021312976, 4.5397868702390376e-5]]
        for (i, result) in enumerate(results)
            for normal_fam in (NormalMeanVariance, NormalMeanPrecision, NormalWeightedMeanPrecision)
                q_in_adj = convert(normal_fam, q_in[i])
                @test_rules [check_type_promotion = true, atol = [Float64 => 1e-5]] Sigmoid(:out, Marginalisation) [(
                    input = (q_in = q_in_adj, q_ζ = PointMass(2.0)), output = Categorical(result)
                )]
            end
        end
    end
end
