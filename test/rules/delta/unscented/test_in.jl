module RulesDeltaUTInTest

using Test
using ReactiveMP
import ReactiveMP: @test_rules

# TODO: with_float_conversions = true breaks

# g: single input, single output
g(x::Float64) = x^2 - 5.0
g(x::Vector{Float64}) = x .^ 2 .- 5.0
g_inv(y::Float64) = sqrt(y + 5.0)
g_inv(y::Vector{Float64}) = sqrt.(y .+ 5.0)

# h: multiple input, single output
h(x::Float64, y::Float64) = x^2 - y
h(x::Vector{Float64}, y::Vector{Float64}) = x .^ 2 .- y
h(x::Float64, y::Vector{Float64}) = x^2 .- y
h_inv_x(z::Float64, y::Float64) = sqrt(z + y)
h_inv_x(z::Vector{Float64}, y::Vector{Float64}) = sqrt.(z .+ y)

# g provided in a similar syntax like the N parameter in normal_mixture/test_out.jl
# normal_mixture is the only example with this syntax (that has a test; gamma_mixture is another candidate but ∄ test)

@testset "rules:Delta:unscented:in" begin
    @testset "Belief Propagation: f(x) (m_ins::NormalMeanVariance, *)" begin
        @test_rules [with_float_conversions = false] DeltaFn{g}(:in, Marginalisation) [
            (
                input = (m_out = ManyOf(NormalMeanVariance(2.0, 3.0),NormalMeanVariance(2.0, 1.0)), meta = DeltaUnscented()),
                output = NormalMeanVariance(2.499999999868301, 0.3125000002253504)
            ),
            (
                input = (m_out = ManyOf(NormalMeanVariance(2.0, 3.0),NormalMeanVariance(2.0, 1.0)), meta = DeltaUnscented(alpha = 1.0)),
                output = NormalMeanVariance(2.5, 0.3125)
            )
        ]
    end
    @testset "Belief Propagation: f(x): (m_ins::MvNormalMeanCovariance, *)" begin
        @test_rules [with_float_conversions = false] DeltaFn{g}(:in, Marginalisation) [
            (
                input = (m_out = ManyOf(MvNormalMeanCovariance([2.0], [3.0]),NormalMeanVariance([2.0], [1.0])), meta = DeltaUnscented()),
                output = MvNormalMeanCovariance([2.499999999868301], [0.31250000021807445])
            ),
            (
                input = (m_out = ManyOf(MvNormalMeanCovariance([2.0], [3.0]),NormalMeanVariance([2.0], [1.0])), meta = DeltaUnscented(alpha = 1.0)),
                output = MvNormalMeanCovariance([2.5], [0.3125])
            )
        ]
    end

    @test ruleSPDeltaUTInGX(h, 1, Message(Univariate, Gaussian{Moments}, m=2.0, v=3.0), Message(Univariate, Gaussian{Moments}, m=2.0, v=1.0), Message(Univariate, Gaussian{Moments}, m=5.0, v=1.0)) == Message(Univariate, Gaussian{Canonical}, xi=6.666665554127903, w=2.6666662216903676)
    @test ruleSPDeltaUTInGX(h, 1, Message(Multivariate, Gaussian{Moments}, m=[2.0], v=[3.0]), Message(Multivariate, Gaussian{Moments}, m=[2.0], v=[1.0]), Message(Multivariate, Gaussian{Moments}, m=[5.0], v=[1.0])) == Message(Multivariate, Gaussian{Canonical}, xi=[6.666665554127903], w=[2.666666221690368])
    
    @testset "Belief Propagation: f(x,y) (m_ins::NormalMeanVariance, *)" begin
        @test_rules [with_float_conversions = false] DeltaFn{h}(:in, Marginalisation) [
            (
            input = (m_out = ManyOf(NormalMeanVariance(2.0, 3.0), NormalMeanVariance(2.0, 1.0),NormalMeanVariance(5.0, 1.0)), meta = DeltaUnscented(inverse = nothing)),
            output = NormalMeanPrecision(6.666665554127903, 2.6666662216903676)
        )
        ]
    end

    @testset "Belief Propagation: f(x,y) (m_ins::MvNormalMeanCovariance, *)" begin
        @test_rules [with_float_conversions = false] DeltaFn{h}(:in, Marginalisation) [
            (
            input = (m_out = ManyOf(MvNormalMeanCovariance([2.0], [3.0]), MvNormalMeanCovariance([2.0], [1.0]),MvNormalMeanCovariance([5.0], [1.0])), meta = DeltaExtended()),
            output = MvNormalMeanPrecision([6.666665554127903], [2.666666221690368])
        )
        ]
    end

end # testset
end # module
