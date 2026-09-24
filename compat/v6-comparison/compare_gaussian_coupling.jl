# GaussianCouplingMessagePassingRules compared with v6's GaussianCoupling on identical inputs:
# the messages, the joint and the average energy.
#
#   julia --startup-file=no --project=compat/v6-comparison compat/v6-comparison/compare_gaussian_coupling.jl

include(joinpath(@__DIR__, "V6Oracle.jl"))

using .V6Oracle, Test
using ExponentialFamily, BayesBase
using MessagePassingRulesBase, MessagePassingRulesTestUtils
import GaussianCouplingMessagePassingRules as GC
import ReactiveMP

const MESSAGES = [NormalMeanVariance(2.0, 3.0), NormalMeanPrecision(-1.0, 4.0), NormalWeightedMeanPrecision(1.0, 2.0)]
const COUPLINGS = [PointMass(1.0), PointMass(-0.5), PointMass(0.0), PointMass(3.0)]

@testset "GaussianCouplingMessagePassingRules against v6" begin
    for (target, other) in ((:out, :in), (:in, :out)), m in MESSAGES, a in COUPLINGS
        inputs = NamedTuple{(other,)}((m,))
        v7 = call_message_update_rule(GC.GaussianCoupling, target; m = inputs, q = (a = a,))
        v6, _ = v6_message_update(ReactiveMP.GaussianCoupling, target, inputs, (a = a,))
        @test compare_with_reference("GaussianCoupling:$target", v7, v6; node = "GaussianCoupling", target = ":$target").outcome === :agree
    end
    for m_out in MESSAGES, m_in in MESSAGES, a in COUPLINGS
        v7 = call_marginal_update_rule(GC.GaussianCoupling, (:out, :in); m = (out = m_out, in = m_in), q = (a = a,))
        v6 = v6_marginal_update(ReactiveMP.GaussianCoupling, (:out, :in), (out = m_out, in = m_in), (a = a,))
        @test compare_with_reference("GaussianCoupling:joint", v7, v6; node = "GaussianCoupling", target = "(:out, :in)").outcome === :agree
    end
    for joint in (MvNormalMeanCovariance([1.0, 2.0], [2.0 0.5; 0.5 3.0]), MvNormalMeanPrecision([1.0, -1.0], [2.0 0.0; 0.0 4.0])), a in COUPLINGS
        v7 = call_average_energy(GC.GaussianCoupling; clusters = ((:out, :in) => joint,), q = (a = a,))
        v6 = v6_average_energy(ReactiveMP.GaussianCoupling, (a = a,), ((:out, :in) => joint,))
        @test compare_with_reference("GaussianCoupling:energy", v7, v6; node = "GaussianCoupling", target = "energy").outcome === :agree
    end
end
