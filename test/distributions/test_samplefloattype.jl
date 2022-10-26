module ReactiveMPsamplefloattypeTest

using Test
using ReactiveMP
using MacroTools

import ReactiveMP: rule_macro_parse_on_tag, rule_macro_parse_fn_args, call_rule_macro_parse_fn_args

@testset "samplefloattype" begin
    @testset "distributions" begin
        @testset "Wishart" begin
            for i in 1:10
                @test ReactiveMP.samplefloattype(Wishart(i, diageye(i))) === Float64
            end
        end
        @testset "MvNormalMeanPrecision" begin
            @test ReactiveMP.samplefloattype(MvNormalMeanPrecision([1.0, 1.0])) === Float64
            @test ReactiveMP.samplefloattype(MvNormalMeanPrecision([1.0, 1.0], [1.0, 1.0])) === Float64
            @test ReactiveMP.samplefloattype(MvNormalMeanPrecision([1, 1])) === Float64
            @test ReactiveMP.samplefloattype(MvNormalMeanPrecision([1, 1], [1, 1])) === Float64
            @test ReactiveMP.samplefloattype(MvNormalMeanPrecision([1.0f0, 1.0f0])) === Float32
            @test ReactiveMP.samplefloattype(MvNormalMeanPrecision([1.0f0, 1.0f0], [1.0f0, 1.0f0])) === Float32
        end

        @testset "NormalMeanPrecision" begin
            @test ReactiveMP.samplefloattype(NormalMeanPrecision()) === Float64
            @test ReactiveMP.samplefloattype(NormalMeanPrecision(0.0)) === Float64
            @test ReactiveMP.samplefloattype(NormalMeanPrecision(0.0, 1.0)) === Float64
            @test ReactiveMP.samplefloattype(NormalMeanPrecision(0)) === Float64
            @test ReactiveMP.samplefloattype(NormalMeanPrecision(0, 1)) === Float64
            @test ReactiveMP.samplefloattype(NormalMeanPrecision(0.0, 1)) === Float64
            @test ReactiveMP.samplefloattype(NormalMeanPrecision(0, 1.0)) === Float64
            @test ReactiveMP.samplefloattype(NormalMeanPrecision(0.0f0)) === Float32
            @test ReactiveMP.samplefloattype(NormalMeanPrecision(0.0f0, 1.0f0)) === Float32
            @test ReactiveMP.samplefloattype(NormalMeanPrecision(0.0f0, 1.0)) === Float64
        end

        @testset "NormalMeanVariance" begin
            @test ReactiveMP.samplefloattype(NormalMeanVariance()) === Float64
            @test ReactiveMP.samplefloattype(NormalMeanVariance(0.0)) === Float64
            @test ReactiveMP.samplefloattype(NormalMeanVariance(0.0, 1.0)) === Float64
            @test ReactiveMP.samplefloattype(NormalMeanVariance(0)) === Float64
            @test ReactiveMP.samplefloattype(NormalMeanVariance(0, 1)) === Float64
            @test ReactiveMP.samplefloattype(NormalMeanVariance(0.0, 1)) === Float64
            @test ReactiveMP.samplefloattype(NormalMeanVariance(0, 1.0)) === Float64
            @test ReactiveMP.samplefloattype(NormalMeanVariance(0.0f0)) === Float32
            @test ReactiveMP.samplefloattype(NormalMeanVariance(0.0f0, 1.0f0)) === Float32
            @test ReactiveMP.samplefloattype(NormalMeanVariance(0.0f0, 1.0)) === Float64
        end

        @testset "NormalWeightedMeanPrecision" begin
            @test ReactiveMP.samplefloattype(NormalWeightedMeanPrecision()) === Float64
            @test ReactiveMP.samplefloattype(NormalWeightedMeanPrecision(0.0)) === Float64
            @test ReactiveMP.samplefloattype(NormalWeightedMeanPrecision(0.0, 1.0)) === Float64
            @test ReactiveMP.samplefloattype(NormalWeightedMeanPrecision(0)) === Float64
            @test ReactiveMP.samplefloattype(NormalWeightedMeanPrecision(0, 1)) === Float64
            @test ReactiveMP.samplefloattype(NormalWeightedMeanPrecision(0.0, 1)) === Float64
            @test ReactiveMP.samplefloattype(NormalWeightedMeanPrecision(0, 1.0)) === Float64
            @test ReactiveMP.samplefloattype(NormalWeightedMeanPrecision(0.0f0)) === Float32
            @test ReactiveMP.samplefloattype(NormalWeightedMeanPrecision(0.0f0, 1.0f0)) === Float32
            @test ReactiveMP.samplefloattype(NormalWeightedMeanPrecision(0.0f0, 1.0)) === Float64
        end
    end

    @testset "samplelist" begin
        @test ReactiveMP.samplefloattype(SampleList([1, 1.0])) === Float64
        @test ReactiveMP.samplefloattype(SampleList([[1, 1.0], [1.0, 1.0]])) === Float64
        @test ReactiveMP.samplefloattype(SampleList([[1 1; 1.0 1], [1.0 1; 1.0 1]])) === Float64
    end
end

end
