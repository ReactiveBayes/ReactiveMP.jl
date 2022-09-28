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
    @testset "Belief Propagation: f(x) (m_ins::NormalMeanCovariance, meta::DeltaExtended) (with known inverse)" begin
        @test_rules [with_float_conversions = false] DeltaFn{g}((:in, k = 1), Marginalisation) [
            (
            input = (m_out = NormalMeanVariance(0.0, 1.0), m_ins = nothing, meta = DeltaExtended(inverse = g_inv)),
            output = NormalMeanVariance(2.23606797749979, 0.05) # TODO: double check this
        )
        ]
    end
end # testset
end # module
