
@testitem "TensorDirichletNode" begin
    using ReactiveMP, Random, BayesBase, ExponentialFamily, Distributions, StableRNGs

    @testset "AverageEnergy" begin
        begin
            rng = StableRNG(123456)
            for i in 1:100
                α     = rand(rng, 2, 2)
                a     = rand(rng, 2, 2)
                q_out = TensorDirichlet(α)
                q_a   = PointMass(a)

                marginals = (Marginal(q_out, false, false, nothing), Marginal(q_a, false, false, nothing))
                avg_energy = score(AverageEnergy(), TensorDirichlet, Val{(:out, :a)}(), marginals, nothing)

                q_out = MatrixDirichlet(α)
                q_a   = PointMass(a)

                marginals = (Marginal(q_out, false, false, nothing), Marginal(q_a, false, false, nothing))
                avg_energy_matrix = score(AverageEnergy(), MatrixDirichlet, Val{(:out, :a)}(), marginals, nothing)

                @test avg_energy ≈ avg_energy_matrix
            end
        end

        begin
            for rank in 3:5
                for dim in 2:5
                    for i in 1:100
                        dims = ntuple(d -> dim, rank)
                        α = rand(rng, dims...)
                        a = rand(rng, dims...)

                        q_out = TensorDirichlet(α)
                        q_a   = PointMass(a)

                        marginals = (Marginal(q_out, false, false, nothing), Marginal(q_a, false, false, nothing))
                        avg_energy = score(AverageEnergy(), TensorDirichlet, Val{(:out, :a)}(), marginals, nothing)

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
end
