# ProbitMessagePassingRules compared with v6's Probit on identical inputs: the messages under
# expectation propagation and under the default algorithm, the joint and the average energy.
#
#   julia --startup-file=no --project=compat/v6-comparison compat/v6-comparison/compare_probit.jl

include(joinpath(@__DIR__, "V6Oracle.jl"))

using .V6Oracle, Test
using ExponentialFamily, BayesBase, Distributions
using MessagePassingRulesBase, MessagePassingRulesTestUtils, ProbitMessagePassingRules
import ReactiveMP

const META = ReactiveMP.ProbitMeta(32)

# v6's energy took log(Φ(x)), which underflows to -Inf at the far Gauss–Hermite points of a wide
# q(in): Inf, or NaN for a point-mass output. The port computes log Φ directly.
const ENERGY_UNDERFLOW = "v6's Probit energy computes log(normcdf(x)), which underflows at far cubature points, giving Inf or NaN; the port uses normlogcdf"
const CAVITIES = [NormalMeanVariance(1.0, 0.5), NormalMeanPrecision(-0.5, 3.0), NormalWeightedMeanPrecision(2.0, 2.0), NormalMeanVariance(-2.0, 4.0)]
const OUTPUTS = [PointMass(1.0), PointMass(0.0), Bernoulli(0.8), Bernoulli(0.3), Bernoulli(0.5)]

@testset "ProbitMessagePassingRules against v6" begin
    for algorithm in (ProbitEP(), DefaultAlgorithm()), m_in in (CAVITIES..., PointMass(1.0), PointMass(-0.5))
        v7 = call_message_update_rule(Probit, :out; m = (in = m_in,), algorithm)
        v6, _ = v6_message_update(ReactiveMP.Probit, :out, (in = m_in,), NamedTuple(); meta = META)
        @test compare_with_reference("Probit:out", v7, v6; node = "Probit", target = ":out").outcome === :agree
    end
    for m_out in OUTPUTS, m_in in CAVITIES
        v7 = call_message_update_rule(Probit, :in; m = (out = m_out, in = m_in))
        v6, _ = v6_message_update(ReactiveMP.Probit, :in, (out = m_out, in = m_in), NamedTuple(); meta = META)
        @test compare_with_reference("Probit:in:EP", v7, v6; node = "Probit", target = ":in").outcome === :agree
    end
    # Belief propagation towards `in` is a log-density, compared at points.
    for m_out in OUTPUTS
        v7 = call_message_update_rule(Probit, :in; m = (out = m_out,), algorithm = DefaultAlgorithm())
        v6, _ = v6_message_update(ReactiveMP.Probit, :in, (out = m_out,), NamedTuple(); meta = META)
        @test all(z -> logpdf(v7, z) ≈ logpdf(v6, z), (-3.0, -0.5, 0.0, 0.4, 2.5))
    end
    for p in (1.0, 0.0), m_in in CAVITIES
        v7 = call_marginal_update_rule(Probit, (:out, :in); m = (out = PointMass(p), in = m_in))
        v6 = v6_marginal_update(ReactiveMP.Probit, (:out, :in), (out = PointMass(p), in = m_in), NamedTuple(); meta = META)
        @test compare_with_reference("Probit:joint", v7, FactorizedCluster(v6_cluster_blocks(ReactiveMP.Probit, v6)...); node = "Probit", target = "(:out, :in)").outcome === :agree
    end
    for q_out in OUTPUTS, q_in in CAVITIES, p in (32, 100)
        v7 = call_average_energy(Probit; q = (out = q_out, in = q_in), algorithm = ProbitEP(p))
        v6 = v6_average_energy(ReactiveMP.Probit, (out = q_out, in = q_in); meta = ReactiveMP.ProbitMeta(p))
        declared = isfinite(v6) ? DeclaredDisagreement[] : [DeclaredDisagreement("Probit:energy"; kind = :correction, reasoning = ENERGY_UNDERFLOW)]
        outcome = compare_with_reference("Probit:energy", v7, v6; inputs = (q_out = q_out, q_in = q_in, p = p), node = "Probit", target = "energy", declared).outcome
        @test outcome === (isfinite(v6) ? :agree : :correction)
        @test isfinite(v7)
    end
end
