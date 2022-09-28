module RulesDeltaETMarginalsTest

using Test
using ReactiveMP
import ReactiveMP: @test_marginalrules

# TODO: with_float_conversions = true breaks

# g: single input, single output
g(x::Number) = x^2 - 5.0
g(x::Vector) = x .^ 2 .- 5.0
g_inv(y::Number) = sqrt(y + 5.0)
g_inv(y::Vector) = sqrt.(y .+ 5.0)

# h: multiple inut, single output
h(x::Number, y::Number) = x^2 - y
h(x::Vector, y::Vector) = x .^ 2 .- y
h_inv_x(z::Number, y::Number) = sqrt(z + y)
h_inv_x(z::Vector, y::Vector) = sqrt.(z .+ y)

@testset "rules:Delta:extended:marginals" begin
    @testset "Marginal: f(x) (m_ins::NormalMeanCovariance, meta.inverse::Nothing)" begin
        # ForneyLab:test_delta_extended:MDeltaEInGX 1
        @test_marginalrules [with_float_conversions = false] DeltaFn{h}(:ins) [
            (
            input = (
                m_out = NormalMeanVariance(2.0, 3.0),
                m_ins = ManyOf(NormalMeanVariance(2.0, 1.0), NormalMeanVariance(5.0, 1.0)),
                meta  = DeltaExtended(inverse = nothing)
            ),
            #output = DeltaMarginal(MvNormalMeanCovariance([2.3636363470614055, 4.9090909132334355], [0.2727273058237252 0.1818181735464949; 0.18181817354649488 0.9545454566127697]), Any[(), ()])
            output = MvNormalMeanCovariance(
                [2.3636363470614055, 4.9090909132334355],
                [0.2727273058237252 0.1818181735464949; 0.18181817354649488 0.9545454566127697]
            )
        )
        ]
    end
end # testset
end # module
