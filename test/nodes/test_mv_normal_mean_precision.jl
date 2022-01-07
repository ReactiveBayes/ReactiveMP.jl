module MvNormalMeanPrecisionNodeTest

using Test
using ReactiveMP
using Random

import ReactiveMP: make_node

@testset "MvNormalMeanPrecisionNode" begin

    @testset "Creation" begin
        node = make_node(MvNormalMeanPrecision)

        @test functionalform(node)          === MvNormalMeanPrecision
        @test sdtype(node)                  === Stochastic()
        @test name.(interfaces(node))       === (:out, :μ, :Λ)
        @test factorisation(node)           === ((1, 2, 3), )
        @test localmarginalnames(node)      === (:out_μ_Λ, )
        @test metadata(node)                === nothing

        node = make_node(MvNormalMeanPrecision, meta = 1)

        @test functionalform(node)          === MvNormalMeanPrecision
        @test sdtype(node)                  === Stochastic()
        @test name.(interfaces(node))       === (:out, :μ, :Λ)
        @test factorisation(node)           === ((1, 2, 3), )
        @test localmarginalnames(node)      === (:out_μ_Λ, )
        @test metadata(node)                === 1

        node = make_node(MvNormalMeanPrecision, factorisation = ((1, ), (2, 3)))

        @test functionalform(node)          === MvNormalMeanPrecision
        @test sdtype(node)                  === Stochastic()
        @test name.(interfaces(node))       === (:out, :μ, :Λ)
        @test factorisation(node)           === ((1, ), (2, 3, ))
        @test localmarginalnames(node)      === (:out, :μ_Λ, )
        @test metadata(node)                === nothing
    end

    @testset "AverageEnergy" begin 

        begin 
            q_out = PointMass([ 1.0, 1.0 ])
            q_μ   = MvNormalMeanPrecision([ 1.0, 1.0 ], [ 1.0 0.0; 0.0 1.0 ])
            q_Λ   = Wishart(4, [ 2.0 0.0; 0.0 2.0 ])

            for N in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision), G in (Wishart, )
                marginals = (Marginal(q_out, false, false), Marginal(convert(N, q_μ), false, false), Marginal(convert(G, q_Λ), false, false))
                @test score(AverageEnergy(), MvNormalMeanPrecision, Val{(:out, :μ, :Λ)}, marginals, nothing) ≈ 6.721945550750932
            end
        end

        # begin 
        #     q_out = NormalMeanPrecision(1.0, 1.0)
        #     q_μ   = NormalMeanPrecision(1.0, 1.0)
        #     q_τ   = GammaShapeRate(1.5, 1.5)

        #     for N in (NormalMeanPrecision, NormalMeanVariance, NormalWeightedMeanPrecision), G in (GammaShapeRate, GammaShapeScale)
        #         marginals = (Marginal(q_out, false, false), Marginal(convert(N, q_μ), false, false), Marginal(convert(G, q_τ), false, false))
        #         @test score(AverageEnergy(), NormalMeanPrecision, Val{(:out, :μ, :τ)}, marginals, nothing) ≈ 2.1034261002694663
        #     end
        # end

        # begin 
        #     q_out = PointMass(0.956629)
        #     q_μ   = NormalMeanPrecision(0.255332, 0.762870)
        #     q_τ   = GammaShapeRate(0.93037, 0.79312)

        #     for N in (NormalMeanPrecision, NormalMeanVariance, NormalWeightedMeanPrecision), G in (GammaShapeRate, GammaShapeScale)
        #         marginals = (Marginal(q_out, false, false), Marginal(convert(N, q_μ), false, false), Marginal(convert(G, q_τ), false, false))
        #         @test score(AverageEnergy(), NormalMeanPrecision, Val{(:out, :μ, :τ)}, marginals, nothing) ≈ 2.209338084063204
        #     end
        # end

        # begin 
        #     q_out = NormalMeanPrecision(0.148725, 0.483501)
        #     q_μ   = NormalMeanPrecision(0.992776, 0.545851)
        #     q_τ   = GammaShapeRate(0.309396, 0.343814)

        #     for N in (NormalMeanPrecision, NormalMeanVariance, NormalWeightedMeanPrecision), G in (GammaShapeRate, GammaShapeScale)
        #         marginals = (Marginal(q_out, false, false), Marginal(convert(N, q_μ), false, false), Marginal(convert(G, q_τ), false, false))
        #         @test score(AverageEnergy(), NormalMeanPrecision, Val{(:out, :μ, :τ)}, marginals, nothing) ≈ 4.155913074139921
        #     end
        # end

        # begin 
        #     q_out_μ = MvNormalMeanPrecision([0.2818402997601115, 0.0847764277628964], [0.7059042678955475 0.3595204552322394; 0.3595204552322394 0.22068491824258746])
        #     q_τ     = GammaShapeRate(0.49074414, 0.4071772)

        #     for N in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision), G in (GammaShapeRate, GammaShapeScale)
        #         marginals = (Marginal(convert(N, q_out_μ), false, false), Marginal(convert(G, q_τ), false, false))
        #         @test score(AverageEnergy(), NormalMeanPrecision, Val{(:out_μ, :τ)}, marginals, nothing) ≈ 38.88138702883774
        #     end
        # end

        # begin 
        #     q_out_μ = MvNormalMeanPrecision([0.8378350736808462, 0.41494396892699026], [1.0074451742986652 0.4369298270351709; 0.4369298270351709 0.19572138039218784])
        #     q_τ     = GammaShapeRate(0.435485, 0.5269575)

        #     for N in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision), G in (GammaShapeRate, GammaShapeScale)
        #         marginals = (Marginal(convert(N, q_out_μ), false, false), Marginal(convert(G, q_τ), false, false))
        #         @test score(AverageEnergy(), NormalMeanPrecision, Val{(:out_μ, :τ)}, marginals, nothing) ≈ 138.6947657738283
        #     end
        # end

    end

end
end
