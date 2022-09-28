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

# h: multiple inut, single output
h(x::Float64, y::Float64) = x^2 - y
h(x::Vector{Float64}, y::Vector{Float64}) = x .^ 2 .- y
h(x::Float64, y::Vector{Float64}) = x^2 .- y
h_inv_x(z::Float64, y::Float64) = sqrt(z + y)
h_inv_x(z::Vector{Float64}, y::Vector{Float64}) = sqrt.(z .+ y)

# g provided in a similar syntax like the N parameter in normal_mixture/test_out.jl
# normal_mixture is the only example with this syntax (that has a test; gamma_mixture is another candidate but ∄ test)
@testset "rules:Delta:unscented:in" begin
    # ForneyLab:test_delta_unscented:SPDeltaUTIn1GG 1-2
    @testset "Belief Propagation: f(x) (m_ins::NormalMeanCovariance, meta.inverse::Nothing)" begin
        @test_rules [with_float_conversions = false] DeltaFn{g}((:in, k), Marginalisation) [
            (
            input = (m_out = ManyOf(NormalMeanVariance(2.0, 3.0)), m_ins = ManyOf(NormalMeanVariance(2.0, 1.0)), meta = DeltaUnscented(inverse = nothing)),
            output = NormalMeanVariance(2.499999999868301, 0.3125000002253504)
        )
        ]
    end
end # testset
end # module
