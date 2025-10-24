@testitem "rules:Sigmoid:zeta" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions
    using StatsFuns: logistic

    import ReactiveMP: @test_rules

    @testset "Mean Field: (q_out::Any, q_in::UnivariateNormalDistributionsFamily)" begin
        q_in = [NormalMeanVariance(0.0, 1.0), NormalMeanVariance(-1.0, 1.0), NormalMeanVariance(10.0, 1.0)]
        results = [1.0, 1.4142135623730951, 10.04987562112089]
        for (i, result) in enumerate(results)
            for normal_fam in (NormalMeanVariance, NormalMeanPrecision, NormalWeightedMeanPrecision)
                q_in_adj = convert(normal_fam, q_in[i])
                @test_rules [check_type_promotion = false, atol = [Float64 => 1e-5]] Sigmoid(:ζ, Marginalisation) [(
                    input = (q_out = 2.0, q_in = q_in_adj), output = PointMass(result)
                )]
            end
        end
    end
end
