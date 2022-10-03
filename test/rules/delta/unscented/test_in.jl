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
h_inv_z(x::Float64, y::Float64) = x^2 - y
h_inv_z(x::Vector{Float64}, y::Vector{Float64}) = x .^ 2 .- y

# g provided in a similar syntax like the N parameter in normal_mixture/test_out.jl
# normal_mixture is the only example with this syntax (that has a test; gamma_mixture is another candidate but ∄ test)

@testset "rules:Delta:unscented:in" begin
    @testset "Single input with known inverse" begin
        @test_rules [with_float_conversions = false] DeltaFn{g}((:in, k = 1), Marginalisation) [
            (
                input = (m_out = NormalMeanVariance(2.0, 3.0), m_ins = nothing, meta = DeltaUnscented(inverse = g_inv)),
                output = NormalMeanVariance(2.6255032138433307, 0.10796282966583703)
            ),
            (
                input = (m_out = MvNormalMeanCovariance([2.0], [3.0;;]), m_ins = nothing, meta = DeltaUnscented(inverse = g_inv)),
                output = MvNormalMeanCovariance([2.6255032138433307], [0.10796282966583703;;])
            )
        ]
    end
    @testset "Multiple input with known inverse" begin
        @test_rules [with_float_conversions = false] DeltaFn{h}((:in, k = 1), Marginalisation) [
            (
                input = (
                    m_out = NormalMeanVariance(2.0, 3.0),
                    m_ins = ManyOf(NormalMeanVariance(5.0, 1.0)),
                    meta = DeltaUnscented(inverse = (h_inv_x, h_inv_z))
                ),
                output = NormalMeanVariance(2.6187538476660848, 0.14431487274498522)
            ),
            (
                input = (m_out = MvNormalMeanCovariance([2.0], [3.0;;]),
                    m_ins = ManyOf(MvNormalMeanCovariance([5.0], [1.0;;])), meta = DeltaUnscented(inverse = (h_inv_x, h_inv_z))),
                output = MvNormalMeanCovariance([2.6187538476660848], [0.14431487274498522;;])
            )
        ]
        @test_rules [with_float_conversions = false] DeltaFn{h}((:in, k = 2), Marginalisation) [
            (
                input = (
                    m_out = NormalMeanVariance(2.0, 1.0),
                    m_ins = ManyOf(NormalMeanVariance(3.0, 1.0)),
                    meta = DeltaUnscented(inverse = (h_inv_x, h_inv_z))
                ),
                output = NormalMeanVariance(2.0000000002328306, 19.00000100088073)
            ),
            (
                input = (m_out = MvNormalMeanCovariance([2.0], [1.0]),
                    m_ins = ManyOf(MvNormalMeanCovariance([3.0], [1.0])), meta = DeltaUnscented(inverse = (h_inv_x, h_inv_z))),
                output = MvNormalMeanCovariance([2.0000000002328306], [19.00000100088073;;])
            )
        ]
    end

    @testset "Single input with unknown inverse" begin
        @test_rules [with_float_conversions = false, atol = 1e-3] DeltaFn{h}((:in, k = 1), Marginalisation) [
            (
                input = (
                    q_ins = DeltaMarginal(MvNormalMeanCovariance(ones(2), [1.0 0.1; 0.1 1.0]), [(), ()]),
                    m_in = NormalMeanVariance(5.0, 10.0),
                    meta = DeltaUnscented()
                ),
                output = NormalWeightedMeanPrecision(0.5, 0.9)
            ),
            (
                input = (q_ins = DeltaMarginal(MvNormalMeanCovariance(ones(2), [1.0 0.1; 0.1 1.0]), [(1,), (1,)]),
                    m_in = MvNormalMeanCovariance([5.0], [10.0;;]), meta = DeltaUnscented()),
                output = MvNormalWeightedMeanPrecision([0.5], [0.9;;])
            )
        ]
    end
    @testset "Multiple input with unknown inverse" begin
        @test_rules [with_float_conversions = false] DeltaFn{h}((:in, k = 2), Marginalisation) [
            (
                input = (
                    q_ins = DeltaMarginal(MvNormalMeanCovariance(ones(3), diageye(3)), [(), (), ()]),
                    m_in = NormalMeanVariance(0.0, 10.0),
                    meta = DeltaUnscented()
                ),
                output = NormalWeightedMeanPrecision(1.0, 0.9)
            ),
            (
                input = (
                    q_ins = DeltaMarginal(MvNormalMeanCovariance(ones(3), diageye(3)), [(1,), (2,), ()]),
                    m_in = MvNormalMeanCovariance(zeros(2), 10 * diageye(2)),
                    meta = DeltaUnscented()
                ),
                output = MvNormalWeightedMeanPrecision(ones(2), 0.9 * diageye(2))
            )
        ]
    end
end # testset
end # module
