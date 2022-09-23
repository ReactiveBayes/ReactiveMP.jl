module RulesDeltaOutTest

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_rules

# TODO: with_float_conversions = true breaks

# g: single input, single output
g(x::Float64) = x^2 - 5.0
g(x::Vector{Float64}) = x.^2 .- 5.0
#g_inv(y::Float64) = sqrt(y + 5.0)
#g_inv(y::Vector{Float64}) = sqrt.(y .+ 5.0)


# h: multiple inut, single output
h(x::Float64, y::Float64) = x^2 - y
h(x::Vector{Float64}, y::Vector{Float64}) = x.^2 .- y
h(x::Float64, y::Vector{Float64}) = x^2 .- y
#h_inv_x(z::Float64, y::Float64) = sqrt(z + y)
#h_inv_x(z::Vector{Float64}, y::Vector{Float64}) = sqrt.(z .+ y)

# g provided in a similar syntax like the N parameter in normal_mixture/test_out.jl
# normal_mixture is the only example with this syntax (that has a test; gamma_mixture is another candidate but ∄ test)
@testset "rules:Delta:out" begin
    # ForneyLab:test_delta_unscented:SPDeltaUTOutNG 1-2
    @testset "Belief Propagation: f(x) (m_ins::NormalMeanVariance, *)" begin
        @test_rules [with_float_conversions = false] DeltaFn{g}(:out, Marginalisation) [
            (
                input = (m_ins = ManyOf(NormalMeanVariance(2.0, 3.0)), meta=DeltaUnscented()),
                output = NormalMeanVariance(2.0000000001164153, 66.00000000093132)
            ),
            (
                input = (m_ins = ManyOf(NormalMeanVariance(2.0, 3.0)), meta=DeltaUnscented(alpha=1.0)),
                output = NormalMeanVariance(2.0, 66.0)
            )
        ]
    end

    # ForneyLab:test_delta_unscented:SPDeltaUTOutNG 3-4
    @testset "Belief Propagation: f(x): (m_ins::MvNormalMeanCovariance, *)" begin
        @test_rules [with_float_conversions = false] DeltaFn{g}(:out, Marginalisation) [
            (
                input = (m_ins = ManyOf(MvNormalMeanCovariance([2.0], [3.0])), meta=DeltaUnscented()),
                output = MvNormalMeanCovariance([2.0000000001164153], [66.00000000093132])
            ),
            (
                input = (m_ins = ManyOf(MvNormalMeanCovariance([2.0], [3.0])), meta=DeltaUnscented(alpha=1.0)),
                output = MvNormalMeanCovariance([2.0], [66.0])
            ),
        ]
    end

    # ForneyLab:test_delta_unscented:SPDeltaUTOutNGX 1
    @testset "Belief Propagation: f(x,y) (m_ins::NormalMeanVariance, *)" begin
        @test_rules [with_float_conversions = false] DeltaFn{h}(:out, Marginalisation) [
            (
                input = (m_ins = ManyOf(NormalMeanVariance(2.0, 3.0), NormalMeanVariance(5.0, 1.0)), meta=DeltaUnscented()),
                output = NormalMeanVariance(1.9999999997671694, 67.00000899657607)
            ),
        ]
    end

    # ForneyLab:test_delta_unscented:SPDeltaUTOutNGX 2
    @testset "Belief Propagation: f(x,y) (m_ins::MvNormalMeanCovariance, *)" begin
       @test_rules [with_float_conversions = false] DeltaFn{h}(:out, Marginalisation) [
           (
               input = (m_ins = ManyOf(MvNormalMeanCovariance([2.0], [3.0]), MvNormalMeanCovariance([5.0], [1.0])), meta=DeltaUnscented()),
               output = MvNormalMeanCovariance([1.9999999997671694], [67.00000899657607])
           ),
       ]
    end
end
end
