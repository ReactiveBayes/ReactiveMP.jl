module ReactiveMPSampleTypeTest

using Test
using ReactiveMP
using MacroTools

import ReactiveMP: rule_macro_parse_on_tag, rule_macro_parse_fn_args, call_rule_macro_parse_fn_args

@testset "sampletype" begin
    @testset "distributions" begin
        @testset "MvNormalMeanPrecision" begin
            @test ReactiveMP.sampletype(MvNormalMeanPrecision([1.0, 1.0])) === Float64
            @test ReactiveMP.sampletype(MvNormalMeanPrecision([1.0, 1.0], [1.0, 1.0])) === Float64
            @test ReactiveMP.sampletype(MvNormalMeanPrecision([1, 1])) === Float64
            @test ReactiveMP.sampletype(MvNormalMeanPrecision([1, 1], [1, 1])) === Float64
            @test ReactiveMP.sampletype(MvNormalMeanPrecision([1.0f0, 1.0f0])) === Float32
            @test ReactiveMP.sampletype(MvNormalMeanPrecision([1.0f0, 1.0f0], [1.0f0, 1.0f0])) === Float32
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
        @test ReactiveMP.sampletype(SampleList([[1, 1.0], [1.0, 1.0]])) === Float64
        @test ReactiveMP.sampletype(SampleList([[1 1; 1.0 1], [1.0 1; 1.0 1]])) === Float64
    end
end

end
