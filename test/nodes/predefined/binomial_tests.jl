
@testitem "Binomial Node" begin
    using ReactiveMP, Random, BayesBase, ExponentialFamily

    @testset "AverageEnergy " begin
        q_k = Binomial(2, 0.5) 
        q_n = Binomial(2, 0.5)
        q_p = Beta(1, 1)
        
        marginals = (Marginal(q_n, false, false, nothing), Marginal(q_k, false, false, nothing), Marginal(q_p, false, false, nothing))

        @test score(AverageEnergy(), Binomial, Val{(:n, :k, :p)}(), marginals, nothing) ≈ 0.077 atol = 1e-3 
        
    end

    # @testset "AverageEnergy " begin
    #     q_k = Poisson(2, 0.5) 
    #     q_n = Binomial(2, 0.5)
    #     q_p = Beta(1, 1)
        
    #     marginals = (Marginal(q_n, false, false, nothing), Marginal(q_k, false, false, nothing), Marginal(q_p, false, false, nothing))

    #     @test score(AverageEnergy(), Binomial, Val{(:n, :k, :p)}(), marginals, nothing) ≈ 0.077 atol = 1e-3 
        
    # end

    
end 
