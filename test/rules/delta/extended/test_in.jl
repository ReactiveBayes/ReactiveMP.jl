module RulesDeltaETInTest

using Test
using ReactiveMP
import ReactiveMP: @test_rules

# TODO: with_float_conversions = true breaks

# g: single input, single output
g(x::Real)       = x^2 - 5.0
g(x::Vector)     = x .^ 2 .- 5.0
g_inv(y::Real)   = sqrt(y + 5.0)
g_inv(y::Vector) = sqrt.(y .+ 5.0)

# h: multiple input, single output
h(x::Real, y::Real) = x^2 - y
h(x::Vector, y::Vector) = x .^ 2 .- y
h_inv_x(z::Real, y::Real) = sqrt(z + y)
h_inv_x(z::Vector, y::Vector) = sqrt.(z .+ y)

# g provided in a similar syntax like the N parameter in normal_mixture/test_out.jl
# normal_mixture is the only example with this syntax (that has a test; gamma_mixture is another candidate but ∄ test)

@testset "rules:Delta:extended:in" begin
    @testset "Belief Propagation: f(x) (m_ins::NormalMeanVariance, *)" begin
        @test_rules [with_float_conversions = false] DeltaFn{g}(:in, Marginalisation) [
            (
            input = (m_out = ManyOf(NormalMeanVariance(2.0, 3.0),NormalMeanVariance(2.0, 1.0)), meta = DeltaExtended(inverse = nothing)),
            output = NormalMeanPrecision(14.666666666666666, 5.333333333333333)
        )
        ]
    end

    @testset "Belief Propagation: f(x): (m_ins::MvNormalMeanCovariance, *)" begin
        @test_rules [with_float_conversions = false] DeltaFn{g}(:in, Marginalisation) [
            (
            input = (m_out = ManyOf(MvNormalMeanCovariance([2.0], [3.0]),MvNormalMeanCovariance([2.0], [1.0])), meta = DeltaExtended(inverse = nothing)),
            output = MvNormalMeanPrecision([14.666666666666666], [5.333333333333335])
        )
        ]
    end

    @testset "Belief Propagation: f(x,y) (m_ins::NormalMeanVariance, *)" begin
        @test_rules [with_float_conversions = false] DeltaFn{h}(:in, Marginalisation) [
            (
            input = (m_out = ManyOf(NormalMeanVariance(2.0, 3.0), NormalMeanVariance(2.0, 1.0),NormalMeanVariance(5.0, 1.0)), meta = DeltaExtended(inverse = nothing)),
            output = NormalMeanPrecision(10.999999999999996, 3.9999999999999982)
        )
        ]
    end

    @testset "Belief Propagation: f(x,y) (m_ins::MvNormalMeanCovariance, *)" begin
        @test_rules [with_float_conversions = false] DeltaFn{h}(:in, Marginalisation) [
            (
            input = (m_out = ManyOf(MvNormalMeanCovariance([2.0], [3.0])), MvNormalMeanCovariance([2.0], [1.0]),MvNormalMeanCovariance([5.0], [1.0])), meta = DeltaExtended(),
            output = MvNormalMeanPrecision([10.999999999999996], [3.9999999999999982])
        )
        ]
    end

   
end # testset
end # module
