# SoftDotMessagePassingRules compared with v6's softdot on identical inputs, at orders 1 to 3: the
# variational messages under mean-field and the structured q(y, x), the joint and the energies.
#
#   julia --startup-file=no --project=compat/v6-comparison compat/v6-comparison/compare_softdot.jl

include(joinpath(@__DIR__, "V6Oracle.jl"))

using .V6Oracle, Test
using ExponentialFamily, BayesBase, Distributions
using MessagePassingRulesBase, MessagePassingRulesTestUtils, SoftDotMessagePassingRules
import ReactiveMP

# A normal over d dimensions, univariate when d is 1.
normal(m, v) = length(m) == 1 ? NormalMeanVariance(only(m), only(v)) : MvNormalMeanCovariance(m, v)
spd(d, s) = [i == j ? s + i : 0.2 / (i + j) for i in 1:d, j in 1:d]
spread(a, b, d) = [a + (b - a) * (k - 1) / max(d - 1, 1) for k in 1:d]

@testset "SoftDotMessagePassingRules against v6" begin
    for d in 1:3
        q_θ = normal(spread(0.5, -0.3, d), spd(d, 1.0))
        q_x = normal(spread(-1.0, 2.0, d), spd(d, 0.5))
        m_x = normal(spread(0.3, 1.1, d), spd(d, 2.0))
        q_y, m_y, q_γ = NormalMeanVariance(1.2, 0.7), NormalMeanVariance(-0.4, 1.5), GammaShapeRate(3.0, 2.0)
        V = [i == j ? 1.0 + i : 0.1 for i in 1:(d + 1), j in 1:(d + 1)]
        joint = MvNormalMeanCovariance(spread(0.4, 1.3, d + 1), V)
        cases = [
            (:y, NamedTuple(), (θ = q_θ, x = q_x, γ = q_γ), ()),
            (:y, (x = m_x,), (θ = q_θ, γ = q_γ), ()),
            (:θ, NamedTuple(), (y = q_y, x = q_x, γ = q_γ), ()),
            (:θ, NamedTuple(), (γ = q_γ,), ((:y, :x) => joint,)),
            (:x, NamedTuple(), (y = q_y, θ = q_θ, γ = q_γ), ()),
            (:x, (y = m_y,), (θ = q_θ, γ = q_γ), ()),
            (:γ, NamedTuple(), (y = q_y, θ = q_θ, x = q_x), ()),
            (:γ, NamedTuple(), (θ = q_θ,), ((:y, :x) => joint,)),
        ]
        for (target, m, q, clusters) in cases
            v7 = call_message_update_rule(SoftDot, target; m, q, clusters)
            v6_q = isempty(clusters) ? q : merge((y_x = last(only(clusters)),), q)
            v6, _ = v6_message_update(ReactiveMP.SoftDot, target, m, v6_q)
            @test compare_with_reference("SoftDot:$target:order $d", v7, v6; node = "SoftDot", target = ":$target").outcome === :agree
        end
        v7 = call_marginal_update_rule(SoftDot, (:y, :x); m = (y = m_y, x = m_x), q = (θ = q_θ, γ = q_γ))
        v6 = v6_marginal_update(ReactiveMP.SoftDot, (:y, :x), (y = m_y, x = m_x), (θ = q_θ, γ = q_γ))
        @test compare_with_reference("SoftDot:joint:order $d", v7, v6; node = "SoftDot", target = "(:y, :x)").outcome === :agree
        v7 = call_average_energy(SoftDot; q = (y = q_y, θ = q_θ, x = q_x, γ = q_γ))
        v6 = v6_average_energy(ReactiveMP.SoftDot, (y = q_y, θ = q_θ, x = q_x, γ = q_γ))
        @test compare_with_reference("SoftDot:energy:meanfield:order $d", v7, v6; node = "SoftDot", target = "energy").outcome === :agree
        v7 = call_average_energy(SoftDot; clusters = ((:y, :x) => joint,), q = (θ = q_θ, γ = q_γ))
        v6 = v6_average_energy(ReactiveMP.SoftDot, (θ = q_θ, γ = q_γ), ((:y, :x) => joint,))
        @test compare_with_reference("SoftDot:energy:structured:order $d", v7, v6; node = "SoftDot", target = "energy").outcome === :agree
    end
end
