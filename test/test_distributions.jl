module ReactiveMPDistributionsTest

using Test
using ReactiveMP
using Distributions

import ReactiveMP: convert_eltype
import ReactiveMP: FactorizedJoint

@testset "Distributions" begin

    @testset "convert_eltype" begin 
        for T in (Float32, Float64, BigFloat)

            @test @inferred(eltype(convert_eltype(T, [ 1.0, 1.0 ]))) === T
            @test @inferred(eltype(convert_eltype(T, [ 1.0 1.0; 1.0 1.0 ]))) === T
            @test @inferred(eltype(convert_eltype(T, 1.0))) === T

        end
    end

    @testset "FactorizedJoint" begin
        vmultipliers = [(NormalMeanPrecision(),), (NormalMeanVariance(), Beta(1.0, 1.0)), (Normal(), Gamma(), MvNormal(zeros(2), diageye(2)))]

        @testset "getindex" begin
            for multipliers in vmultipliers
                product = FactorizedJoint(multipliers)
                @test length(product) === length(multipliers)
                for i in length(multipliers)
                    @test product[i] === multipliers[i]
                end
            end
        end

        @testset "entropy" begin
            for multipliers in vmultipliers
                product = FactorizedJoint(multipliers)
                @test entropy(product) ≈ mapreduce(entropy, +, multipliers)
            end
        end

        @testset "isapprox" begin
            @test FactorizedJoint((NormalMeanVariance(),)) ≈ FactorizedJoint((NormalMeanVariance(),))
            @test !(FactorizedJoint((NormalMeanVariance(),)) ≈ FactorizedJoint((NormalMeanVariance(1, 1),)))

            @test FactorizedJoint((Gamma(1.0, 1.0), NormalMeanVariance(0.0, 1.0))) ≈ FactorizedJoint((Gamma(1.000001, 1.0), NormalMeanVariance(0.0, 1.0000000001))) atol = 1e-5
            @test !(FactorizedJoint((Gamma(1.0, 1.0), NormalMeanVariance(0.0, 1.0))) ≈ FactorizedJoint((Gamma(1.000001, 1.0), NormalMeanVariance(0.0, 5.0000000001))))
            @test !(FactorizedJoint((Gamma(1.0, 2.0), NormalMeanVariance(0.0, 1.0))) ≈ FactorizedJoint((Gamma(1.000001, 1.0), NormalMeanVariance(0.0, 1.0000000001))))
        end
    end
end

end
