# GCVMessagePassingRules compared with v6's GCV on identical inputs: the messages under the
# structured and the mean-field factorisation, the joint of y and x, the average energies, and
# the rules that let the normal nodes take an ExponentialLinearQuadratic message. v6's messages
# towards z, κ and ω are its own ExponentialLinearQuadratic, compared by their coefficients.
#
#   julia --startup-file=no --project=compat/v6-comparison compat/v6-comparison/compare_gcv.jl

include(joinpath(@__DIR__, "V6Oracle.jl"))

using .V6Oracle, Test
using ExponentialFamily, BayesBase, Distributions
using MessagePassingRulesBase, MessagePassingRulesTestUtils, StandardMessagePassingRules, GCVMessagePassingRules
import MessagePassingRulesApproximations
import ReactiveMP

const META = ReactiveMP.GCVMetadata(ReactiveMP.GaussHermiteCubature(20))
# The normal nodes' variational rules correct v6 as Standard does: a q(v) contributes 1/E[1/v].
const NMV_669 = "ReactiveMP.jl#669: v6 takes E[v] for q_v's contribution; the port, as Standard, 1/E[1/v]"
coefficients(d) = (d.a, d.b, d.c, d.d)
v6_elq(d) = ReactiveMP.ExponentialLinearQuadratic(ReactiveMP.GaussHermiteCubature(20), d.a, d.b, d.c, d.d)

const PARAMETERS = [
    (y = NormalMeanVariance(3.0, 1.0), x = NormalMeanVariance(1.0, 2.0), z = NormalMeanVariance(0.5, 0.7), κ = NormalMeanVariance(0.8, 0.4), ω = NormalMeanVariance(1.2, 0.5)),
    (y = NormalMeanVariance(0.4, 0.6), x = NormalMeanPrecision(0.5, 3.0), z = NormalMeanVariance(2.0, 0.3), κ = NormalMeanVariance(1.2, 0.25), ω = NormalMeanVariance(-0.5, 0.8)),
    (y = NormalWeightedMeanPrecision(-1.5, 4.0), x = NormalMeanVariance(2.5, 0.5), z = NormalMeanVariance(-0.75, 1.25), κ = PointMass(0.3), ω = NormalMeanVariance(0.6, 0.15)),
]

@testset "GCVMessagePassingRules against v6" begin
    for p in PARAMETERS
        joint = MvNormalMeanCovariance([mean(p.y), mean(p.x)], [var(p.y) 0.3; 0.3 var(p.x)])
        rest = (z = p.z, κ = p.κ, ω = p.ω)
        # Towards y and x, from a message or a marginal.
        for (target, other) in ((:y, :x), (:x, :y))
            m = NamedTuple{(other,)}((p[other],))
            v7 = call_message_update_rule(GCV, target; m, q = rest)
            v6, _ = v6_message_update(ReactiveMP.GCV, target, m, rest; meta = META)
            @test compare_with_reference("GCV:$target:message", v7, v6; node = "GCV", target = ":$target").outcome === :agree
            v7 = call_message_update_rule(GCV, target; q = merge(m, rest))
            v6, _ = v6_message_update(ReactiveMP.GCV, target, NamedTuple(), merge(m, rest); meta = META)
            @test compare_with_reference("GCV:$target:marginal", v7, v6; node = "GCV", target = ":$target").outcome === :agree
        end
        # Towards z, κ and ω, under both factorisations.
        for target in (:z, :κ, :ω)
            others = NamedTuple{Tuple(filter(!=(target), (:z, :κ, :ω)))}(Tuple(rest[k] for k in (:z, :κ, :ω) if k !== target))
            v7 = call_message_update_rule(GCV, target; q = others, clusters = ((:y, :x) => joint,))
            v6, _ = v6_message_update(ReactiveMP.GCV, target, NamedTuple(), merge((y_x = joint,), others); meta = META)
            @test all(coefficients(v7) .≈ coefficients(v6))
            v7 = call_message_update_rule(GCV, target; q = merge((y = p.y, x = p.x), others))
            v6, _ = v6_message_update(ReactiveMP.GCV, target, NamedTuple(), merge((y = p.y, x = p.x), others); meta = META)
            @test all(coefficients(v7) .≈ coefficients(v6))
            # And its moments, by the same cubature.
            @test all(mean_var(v7) .≈ mean_var(v6))
        end
        v7 = call_marginal_update_rule(GCV, (:y, :x); m = (y = p.y, x = p.x), q = rest)
        v6 = v6_marginal_update(ReactiveMP.GCV, (:y, :x), (y = p.y, x = p.x), rest; meta = META)
        @test compare_with_reference("GCV:joint", v7, v6; node = "GCV", target = "(:y, :x)").outcome === :agree
        if !(p.z isa PointMass)
            v7 = call_average_energy(GCV; clusters = ((:y, :x) => joint,), q = rest)
            v6 = v6_average_energy(ReactiveMP.GCV, rest, ((:y, :x) => joint,); meta = META)
            @test compare_with_reference("GCV:energy:structured", v7, v6; node = "GCV", target = "energy").outcome === :agree
            v7 = call_average_energy(GCV; q = merge((y = p.y, x = p.x), rest))
            v6 = v6_average_energy(ReactiveMP.GCV, merge((y = p.y, x = p.x), rest); meta = META)
            @test compare_with_reference("GCV:energy:meanfield", v7, v6; node = "GCV", target = "energy").outcome === :agree
        end
    end
    # The normal nodes with an ExponentialLinearQuadratic message on `out`.
    elq = call_message_update_rule(GCV, :z; q = (y = NormalMeanVariance(3.0, 1.0), x = NormalMeanVariance(1.0, 2.0), κ = NormalMeanVariance(0.8, 0.4), ω = NormalMeanVariance(1.2, 0.5)))
    μ = NormalMeanVariance(0.5, 2.0)
    for (node, v6_node, parameter, point, marginal) in ((NormalMeanVariance, ReactiveMP.NormalMeanVariance, :v, PointMass(2.0), GammaInverse(3.0, 2.0)), (NormalMeanPrecision, ReactiveMP.NormalMeanPrecision, :τ, PointMass(0.5), GammaShapeRate(3.0, 2.0)))
        for (m, q) in ((NamedTuple{(:out, parameter)}((elq, point)), NamedTuple()), ((out = elq,), NamedTuple{(parameter,)}((marginal,))))
            v7 = call_message_update_rule(node, :μ; m, q)
            v6_m = map(v -> v === elq ? v6_elq(elq) : v, m)
            v6, _ = v6_message_update(v6_node, :μ, v6_m, q)
            corrected = node === NormalMeanVariance && !isempty(q)
            declared = corrected ? [DeclaredDisagreement("$node:μ:elq"; kind = :correction, reasoning = NMV_669)] : DeclaredDisagreement[]
            @test compare_with_reference("$node:μ:elq", v7, v6; node = string(node), target = ":μ", declared).outcome === (corrected ? :correction : :agree)
        end
        v7 = call_marginal_update_rule(node, (:out, :μ, parameter); m = NamedTuple{(:out, :μ, parameter)}((elq, μ, point)))
        v6 = v6_marginal_update(v6_node, (:out, :μ, parameter), NamedTuple{(:out, :μ, parameter)}((v6_elq(elq), μ, point)), NamedTuple())
        @test compare_with_reference("$node:joint:elq", v7, FactorizedCluster(v6_cluster_blocks(v6_node, v6)...); node = string(node), target = "(:out, :μ, :$parameter)").outcome === :agree
        v7 = call_marginal_update_rule(node, (:out, :μ); m = (out = elq, μ = μ), q = NamedTuple{(parameter,)}((marginal,)))
        v6 = v6_marginal_update(v6_node, (:out, :μ), (out = v6_elq(elq), μ = μ), NamedTuple{(parameter,)}((marginal,)))
        declared = node === NormalMeanVariance ? [DeclaredDisagreement("$node:joint:q:elq"; kind = :correction, reasoning = NMV_669)] : DeclaredDisagreement[]
        @test compare_with_reference("$node:joint:q:elq", v7, v6; node = string(node), target = "(:out, :μ)", declared).outcome === (node === NormalMeanVariance ? :correction : :agree)
    end
end
