module MvNormalWeightedMeanPrecisionNodeTest

using Test, ReactiveMP, Random, BayesBase, ExponentialFamily

import ReactiveMP: make_node

@testset "MvNormalWeightedMeanPrecisionNodeTest" begin
    @testset "Creation" begin
        node = make_node(MvNormalWeightedMeanPrecision)

        @test functionalform(node) === MvNormalWeightedMeanPrecision
        @test sdtype(node) === Stochastic()
        @test name.(interfaces(node)) === (:out, :ξ, :Λ)
        @test factorisation(node) === ((1, 2, 3),)
        @test localmarginalnames(node) === (:out_ξ_Λ,)
        @test metadata(node) === nothing

        node = make_node(MvNormalWeightedMeanPrecision, FactorNodeCreationOptions(nothing, 1, nothing))

        @test functionalform(node) === MvNormalWeightedMeanPrecision
        @test sdtype(node) === Stochastic()
        @test name.(interfaces(node)) === (:out, :ξ, :Λ)
        @test factorisation(node) === ((1, 2, 3),)
        @test localmarginalnames(node) === (:out_ξ_Λ,)
        @test metadata(node) === 1

        node = make_node(MvNormalWeightedMeanPrecision, FactorNodeCreationOptions(((1,), (2, 3)), nothing, nothing))

        @test functionalform(node) === MvNormalWeightedMeanPrecision
        @test sdtype(node) === Stochastic()
        @test name.(interfaces(node)) === (:out, :ξ, :Λ)
        @test factorisation(node) === ((1,), (2, 3))
        @test localmarginalnames(node) === (:out, :ξ_Λ)
        @test metadata(node) === nothing
    end

    @testset "AverageEnergy" begin
        begin
            for i in 2:5
                mean_in, L = randn(i), randn(i, i)
                Cov_in = L * L'
                mean_out = randn(i)

                q_out = PointMass(mean_out)
                q_Σ = PointMass(Cov_in)
                q_μ = PointMass(mean_in)

                q_Λ = PointMass(inv(mean(q_Σ)))
                q_ξ = PointMass(mean(q_Λ) * mean(q_μ))

                for N in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision)
                    marginalsξ = (Marginal(q_out, false, false, nothing), Marginal(q_ξ, false, false, nothing), Marginal(q_Λ, false, false, nothing))
                    marginalsμ = (Marginal(q_out, false, false, nothing), Marginal(q_μ, false, false, nothing), Marginal(q_Σ, false, false, nothing))
                    @test score(AverageEnergy(), MvNormalWeightedMeanPrecision, Val{(:out, :ξ, :Λ)}(), marginalsξ, nothing) ≈
                        score(AverageEnergy(), MvNormalMeanCovariance, Val{(:out, :μ, :Σ)}(), marginalsμ, nothing)
                end
            end
        end
        # NOTE: tests for average energy when interfaces are not PointMass are not implemented
    end
end
end
