module RulesDeltaETInTest

using Test
using ReactiveMP
import ReactiveMP: @test_rules

# TODO: with_float_conversions = true breaks

# g: single input, single output
g(x::Number) = x^2 - 5.0
g(x::Vector) = x.^2 .- 5.0
g_inv(y::Number) = sqrt(y + 5.0)
g_inv(y::Vector) = sqrt.(y .+ 5.0)

# h: multiple inut, single output
h(x::Number, y::Number) = x^2 - y
h(x::Vector, y::Vector) = x.^2 .- y
h_inv_x(z::Number, y::Number) = sqrt(z + y)
h_inv_x(z::Vector, y::Vector) = sqrt.(z .+ y)


# g provided in a similar syntax like the N parameter in normal_mixture/test_out.jl
# normal_mixture is the only example with this syntax (that has a test; gamma_mixture is another candidate but ∄ test)
@testset "rules:Delta:extended:in" begin
    # ForneyLab:test_delta_extended:SPDeltaEIn1GG 1-2
    @testset "Belief Propagation: f(x) (m_ins::NormalMeanCovariance, meta.inverse::Nothing)" begin
        @test_rules [with_float_conversions = false] DeltaFn{g}((:in, k), Marginalisation) [
            (
                input = (m_out = ManyOf(NormalMeanVariance(2.0, 3.0)), m_ins =ManyOf(NormalMeanVariance(2.0, 1.0)), meta=DeltaExtended(inverse=nothing)),
                output = NormalMeanVariance(2.499999999868301, 0.3125000002253504)
            ),
        ]
    end
end # testset
end # module
