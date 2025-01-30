@testitem "rules:BinomialPolya:beta" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions, PolyaGammaHybridSamplers

    import ReactiveMP: @test_rules, weightedmean
    import LinearAlgebra: diag

    @testset "Predictive Distribution" begin
        q_x = PointMass([0.1, 0.2])
        q_n = PointMass(5)
        q_β = MvNormalWeightedMeanPrecision([3.0, -1.0], diageye(2))
        
        # Test with default meta
        pred_dist = @call_rule BinomialPolya(:y, Marginalisation) (q_x = q_x, q_n = q_n, q_β = q_β, meta = nothing)
        @test pred_dist isa Binomial
        @test ntrials(pred_dist) == 5
        
        # Test with Monte Carlo sampling
        meta = BinomialPolyaMeta(1000, MersenneTwister(42))
        pred_dist_mc = @call_rule BinomialPolya(:y, Marginalisation) (q_x = q_x, q_n = q_n, q_β = q_β, meta = meta)
        @test pred_dist_mc isa Binomial
        @test ntrials(pred_dist_mc) == 5
        @test 0 < succprob(pred_dist_mc) < 1

        @test pred_dist_mc.p ≈ pred_dist.p atol = 1e-2
    end

end

