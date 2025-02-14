@testitem "DirichletCollectionNode" begin
    using ReactiveMP, Random, BayesBase, ExponentialFamily, Distributions, StableRNGs

    @testset "AverageEnergy" begin
        rng = StableRNG(123456)
        for rank in 3:5
            for dim in 2:5
                for i in 1:100
                    dims = ntuple(d -> dim, rank)
                    α = rand(rng, dims...)
                    a = rand(rng, dims...)

                    q_out = DirichletCollection(α)
                    q_a   = PointMass(a)

                    marginals = (Marginal(q_out, false, false, nothing), Marginal(q_a, false, false, nothing))
                    avg_energy = score(AverageEnergy(), DirichletCollection, Val{(:out, :a)}(), marginals, nothing)

                    q_out = Dirichlet.(eachslice(α, dims = ntuple(d -> d + 1, rank - 1)))
                    q_a   = PointMass.(eachslice(a, dims = ntuple(d -> d + 1, rank - 1)))

                    avg_energy_matrix = 0.0
                    for (dir, a) in zip(q_out, q_a)
                        marginals = (Marginal(dir, false, false, nothing), Marginal(a, false, false, nothing))
                        avg_energy_matrix += score(AverageEnergy(), Dirichlet, Val{(:out, :a)}(), marginals, nothing)
                    end

                    @test avg_energy ≈ avg_energy_matrix
                end
            end
        end
    end
end
