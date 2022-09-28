module RulesDeltaETOutTest

using Test
using ReactiveMP
import ReactiveMP: @test_rules

# TODO: with_float_conversions = true breaks

# g: single input, single output
g(x::Number) = x^2 - 5.0
g(x::Vector) = x .^ 2 .- 5.0

# h: multiple inut, single output
h(x::Number, y::Number) = x^2 - y
h(x::Vector, y::Vector) = x .^ 2 .- y

# g provided in a similar syntax like the N parameter in normal_mixture/test_out.jl
# normal_mixture is the only example with this syntax (that has a test; gamma_mixture is another candidate but ∄ test)
@testset "rules:Delta:extended:out" begin
    # ForneyLab:test_delta_extended:SPDeltaEOutNG 1
    @testset "Belief Propagation: f(x) (m_ins::NormalMeanVariance, *)" begin
        @test_rules [with_float_conversions = false] DeltaFn{g}(:out, Marginalisation) [
            (
            input = (m_ins = ManyOf(NormalMeanVariance(2.0, 3.0)), meta = DeltaExtended(inverse = nothing)),
            output = NormalMeanVariance(-1.0, 48.0)
        )
        ]
    end

    # ForneyLab:test_delta_extended:SPDeltaEOutNG 2
    @testset "Belief Propagation: f(x): (m_ins::MvNormalMeanCovariance, *)" begin
        @test_rules [with_float_conversions = false] DeltaFn{g}(:out, Marginalisation) [
            (
            input = (m_ins = ManyOf(MvNormalMeanCovariance([2.0], [3.0])), meta = DeltaExtended(inverse = nothing)),
            output = MvNormalMeanCovariance([-1.0], [48.0])
        )
        ]
    end

    # ForneyLab:test_delta_extended:SPDeltaEOutNGX 1
    @testset "Belief Propagation: f(x,y) (m_ins::NormalMeanVariance, *)" begin
        @test_rules [with_float_conversions = false] DeltaFn{h}(:out, Marginalisation) [
            (
            input = (m_ins = ManyOf(NormalMeanVariance(2.0, 3.0), NormalMeanVariance(5.0, 1.0)), meta = DeltaExtended(inverse = nothing)),
            output = NormalMeanVariance(-1.0, 49.0)
        )
        ]
    end

    # ForneyLab:test_delta_extended:SPDeltaEOutNGX 2
    @testset "Belief Propagation: f(x,y) (m_ins::MvNormalMeanCovariance, *)" begin
        @test_rules [with_float_conversions = false] DeltaFn{h}(:out, Marginalisation) [
            (
            input = (m_ins = ManyOf(MvNormalMeanCovariance([2.0], [3.0]), MvNormalMeanCovariance([5.0], [1.0])), meta = DeltaExtended()),
            output = MvNormalMeanCovariance([-1.0], [49.0])
        )
        ]
    end
end
end
