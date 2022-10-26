module ReactiveMPSampleTypeTest

using Test
using ReactiveMP
using MacroTools

import ReactiveMP: rule_macro_parse_on_tag, rule_macro_parse_fn_args, call_rule_macro_parse_fn_args

@testset "sampletype" begin
    @testset "distributions" begin
        @testset "Wishart" begin
            for i in 1:10
                @test ReactiveMP.sampletype(Wishart(i, diageye(i))) === Matrix{Float64}
            end
        end
        @testset "MvNormalMeanPrecision" begin
            @test ReactiveMP.sampletype(MvNormalMeanPrecision([1.0, 1.0])) === Vector{Float64}
            @test ReactiveMP.sampletype(MvNormalMeanPrecision([1.0, 1.0], [1.0, 1.0])) === Vector{Float64}
            @test ReactiveMP.sampletype(MvNormalMeanPrecision([1, 1])) === Vector{Float64}
            @test ReactiveMP.sampletype(MvNormalMeanPrecision([1, 1], [1, 1])) === Vector{Float64}
            @test ReactiveMP.sampletype(MvNormalMeanPrecision([1.0f0, 1.0f0])) === Vector{Float32}
            @test ReactiveMP.sampletype(MvNormalMeanPrecision([1.0f0, 1.0f0], [1.0f0, 1.0f0])) === Vector{Float32}
        end

        @testset "NormalMeanPrecision" begin
            @test ReactiveMP.sampletype(NormalMeanPrecision()) === Float64
            @test ReactiveMP.sampletype(NormalMeanPrecision(0.0)) === Float64
            @test ReactiveMP.sampletype(NormalMeanPrecision(0.0, 1.0)) === Float64
            @test ReactiveMP.sampletype(NormalMeanPrecision(0)) === Float64
            @test ReactiveMP.sampletype(NormalMeanPrecision(0, 1)) === Float64
            @test ReactiveMP.sampletype(NormalMeanPrecision(0.0, 1)) === Float64
            @test ReactiveMP.sampletype(NormalMeanPrecision(0, 1.0)) === Float64
            @test ReactiveMP.sampletype(NormalMeanPrecision(0.0f0)) === Float32
            @test ReactiveMP.sampletype(NormalMeanPrecision(0.0f0, 1.0f0)) === Float32
            @test ReactiveMP.sampletype(NormalMeanPrecision(0.0f0, 1.0)) === Float64
        end

        @testset "NormalMeanVariance" begin
            @test ReactiveMP.sampletype(NormalMeanVariance()) === Float64
            @test ReactiveMP.sampletype(NormalMeanVariance(0.0)) === Float64
            @test ReactiveMP.sampletype(NormalMeanVariance(0.0, 1.0)) === Float64
            @test ReactiveMP.sampletype(NormalMeanVariance(0)) === Float64
            @test ReactiveMP.sampletype(NormalMeanVariance(0, 1)) === Float64
            @test ReactiveMP.sampletype(NormalMeanVariance(0.0, 1)) === Float64
            @test ReactiveMP.sampletype(NormalMeanVariance(0, 1.0)) === Float64
            @test ReactiveMP.sampletype(NormalMeanVariance(0.0f0)) === Float32
            @test ReactiveMP.sampletype(NormalMeanVariance(0.0f0, 1.0f0)) === Float32
            @test ReactiveMP.sampletype(NormalMeanVariance(0.0f0, 1.0)) === Float64
        end

        @testset "NormalWeightedMeanPrecision" begin
            @test ReactiveMP.sampletype(NormalWeightedMeanPrecision()) === Float64
            @test ReactiveMP.sampletype(NormalWeightedMeanPrecision(0.0)) === Float64
            @test ReactiveMP.sampletype(NormalWeightedMeanPrecision(0.0, 1.0)) === Float64
            @test ReactiveMP.sampletype(NormalWeightedMeanPrecision(0)) === Float64
            @test ReactiveMP.sampletype(NormalWeightedMeanPrecision(0, 1)) === Float64
            @test ReactiveMP.sampletype(NormalWeightedMeanPrecision(0.0, 1)) === Float64
            @test ReactiveMP.sampletype(NormalWeightedMeanPrecision(0, 1.0)) === Float64
            @test ReactiveMP.sampletype(NormalWeightedMeanPrecision(0.0f0)) === Float32
            @test ReactiveMP.sampletype(NormalWeightedMeanPrecision(0.0f0, 1.0f0)) === Float32
            @test ReactiveMP.sampletype(NormalWeightedMeanPrecision(0.0f0, 1.0)) === Float64
        end
    end

    @testset "samplelist" begin
        @test ReactiveMP.sampletype(SampleList([1, 1.0])) === Float64
        @test ReactiveMP.sampletype(SampleList([[1, 1.0], [1.0, 1.0]])) === Vector{Float64}
        @test ReactiveMP.sampletype(SampleList([[1 1; 1.0 1], [1.0 1; 1.0 1]])) === Matrix{Float64}
    end
end

end
