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
# TODO: inverse for gamma
h_inv_z(x::Float64, y::Float64) = x^2 + y
h_inv_z(x::Vector{Float64}, y::Vector{Float64}) = x .^ 2 .+ y

# g provided in a similar syntax like the N parameter in normal_mixture/test_out.jl
# normal_mixture is the only example with this syntax (that has a test; gamma_mixture is another candidate but ∄ test)

@testset "rules:Delta:unscented:in" begin
    @testset "Single input with known inverse" begin
        @test_rules [with_float_conversions = false] DeltaFn{g}((:in, k = 1), Marginalisation) [
            (
                input = (m_out = NormalMeanVariance(2.0, 3.0), m_ins = nothing, meta = DeltaUnscented(inverse = g_inv)),
                output = NormalMeanVariance(2.499999999868301, 0.3125000002253504)
            ),
            (
                input = (m_out = MvNormalMeanCovariance(ones(2), [1.0 0.0; 0.0 1.0]), m_ins = nothing, meta = DeltaUnscented(inverse = g_inv)),
                output = MvNormalMeanCovariance(zeros(2), [1.0 0.0; 0.0 1.0])
            )
        ]
    end
    @testset "Multiple input with known inverse" begin
        @test_rules [with_float_conversions = false] DeltaFn{h}((:in, k = 1), Marginalisation) [
            (
                input = (
                    m_out = NormalMeanVariance(2.0, 3.0),
                    m_ins = ManyOf(NormalMeanVariance(2.0, 3.0)),
                    meta = DeltaUnscented(inverse = (h_inv_x, h_inv_z))
                ),
                output = NormalMeanVariance(2.499999999868301, 0.3125000002253504)
            ),
            (
                input = (m_out = MvNormalMeanCovariance(ones(2), [1.0 0.0; 0.0 1.0]),
                    m_ins = ManyOf(MvNormalMeanCovariance(ones(2), [1.0 0.0; 0.0 1.0])), meta = DeltaUnscented(inverse = (h_inv_x, h_inv_z))),
                output = MvNormalMeanCovariance(zeros(2), [1.0 0.0; 0.0 1.0])
            )
        ]
        @test_rules [with_float_conversions = false] DeltaFn{h}((:in, k = 2), Marginalisation) [
            (
                input = (
                    m_out = NormalMeanVariance(2.0, 3.0),
                    m_ins = ManyOf(NormalMeanVariance(2.0, 3.0), nothing),
                    meta = DeltaUnscented(inverse = (h_inv_x, h_inv_z))
                ),
                output = NormalMeanVariance(2.499999999868301, 0.3125000002253504)
            ),
            (
                input = (m_out = MvNormalMeanCovariance(ones(2), [1.0 0.0; 0.0 1.0]),
                    m_ins = ManyOf(MvNormalMeanCovariance(ones(2), [1.0 0.0; 0.0 1.0])), meta = DeltaUnscented(inverse = (h_inv_x, h_inv_z))),
                output = MvNormalMeanCovariance(zeros(2), [1.0 0.0; 0.0 1.0])
            )
        ]
    end

    @testset "Single input with unknown inverse" begin
        @test_rules [with_float_conversions = false] DeltaFn{g}((:in, k = 1), Marginalisation) [
            (
                input = (m_out = NormalMeanVariance(2.0, 3.0), m_ins = nothing, meta = DeltaUnscented(inverse = g_inv)),
                output = NormalMeanVariance(2.499999999868301, 0.3125000002253504)
            ),
            (
                input = (m_out = MvNormalMeanCovariance(ones(2), [1.0 0.0; 0.0 1.0]), m_ins = nothing, meta = DeltaUnscented(inverse = g_inv)),
                output = MvNormalMeanCovariance(zeros(2), [1.0 0.0; 0.0 1.0])
            )
        ]
    end
    @testset "Multiple input with unknown inverse" begin
        @test_rules [with_float_conversions = false] DeltaFn{h}((:in, k = 2), Marginalisation) [
            (
                input = (
                    q_ins = DeltaMarginal(MvNormalMeanCovariance(ones(2), [1.0 0.0; 0.0 1.0]), [(), ()]),
                    m_in = NormalMeanVariance(1.0, 1.0),
                    meta = DeltaUnscented()
                ),
                output = NormalMeanVariance(1.0, 1.0)
            ),
            (
                input = (
                    q_ins = DeltaMarginal(MvNormalMeanCovariance(ones(4), diageye(4)), [(), (3,)]),
                    m_in = MvNormalMeanCovariance(ones(3), diageye(3)),
                    meta = DeltaUnscented()
                ),
                output = MvNormalMeanCovariance(ones(3), diageye(3))
            )
        ]
        @test_rules [with_float_conversions = false] DeltaFn{h}((:in, k = 1), Marginalisation) [
            (
                input = (
                    q_ins = DeltaMarginal(MvNormalMeanCovariance(ones(2), [1.0 0.0; 0.0 1.0]), [(), ()]),
                    m_in = NormalMeanVariance(1.0, 1.0),
                    meta = DeltaUnscented()
                ),
                output = NormalMeanVariance(1.0, 1.0)
            ),
            (
                input = (
                    q_ins = DeltaMarginal(MvNormalMeanCovariance(ones(4), diageye(4)), [(), (3,)]),
                    m_in = NormalMeanVariance(1.0, 1.0),
                    meta = DeltaUnscented()
                ),
                output = NormalMeanVariance(1.0, 1.0)
            )
        ]
    end
end # testset
end # module
