# FlowMessagePassingRules compared with v6's Flow on identical inputs: every message rule, under
# Linearization and Unscented, on models with planar and radial coupling flows and a fixed
# permutation, compiled with the same parameters on both sides. v6's marginal rules over `(in,)`
# are not ported, and are compared as what replaces them: the marginal of `in`, the product of
# `m_in` and the rule's message towards `in`.
#
#   julia --startup-file=no --project=compat/v6-comparison compat/v6-comparison/compare_flow.jl

include(joinpath(@__DIR__, "V6Oracle.jl"))

using .V6Oracle, Test
using ExponentialFamily, BayesBase, Distributions, LinearAlgebra
using MessagePassingRulesBase, MessagePassingRulesTestUtils
import FlowMessagePassingRules as F
import MessagePassingRulesApproximations as A
import ReactiveMP

# The same model on both sides: (v6's layers, the port's layers).
function models(kind)
    if kind === :planar
        v6 = ReactiveMP.FlowModel(2, (ReactiveMP.AdditiveCouplingLayer(ReactiveMP.PlanarFlow(); permute = false),))
        v7 = F.FlowModel(2, (F.AdditiveCouplingLayer(F.PlanarFlow(); permute = false),))
    elseif kind === :radial
        v6 = ReactiveMP.FlowModel(2, (ReactiveMP.AdditiveCouplingLayer(ReactiveMP.RadialFlow(); permute = false),))
        v7 = F.FlowModel(2, (F.AdditiveCouplingLayer(F.RadialFlow(); permute = false),))
    else
        v6 = ReactiveMP.FlowModel(
            (
                ReactiveMP.InputLayer(2), ReactiveMP.AdditiveCouplingLayer(ReactiveMP.PlanarFlow(); permute = false),
                ReactiveMP.PermutationLayer(ReactiveMP.PermutationMatrix([2, 1])), ReactiveMP.AdditiveCouplingLayer(ReactiveMP.PlanarFlow(); permute = false),
            )
        )
        v7 = F.FlowModel(
            (
                F.InputLayer(2), F.AdditiveCouplingLayer(F.PlanarFlow(); permute = false),
                F.PermutationLayer(F.PermutationMatrix([2, 1])), F.AdditiveCouplingLayer(F.PlanarFlow(); permute = false),
            )
        )
    end
    params = collect(range(-0.4, 0.6; length = ReactiveMP.nr_params(v6)))
    return ReactiveMP.compile(v6, params), F.compile(v7, params)
end

@testset "FlowMessagePassingRules against v6" begin
    for kind in (:planar, :radial, :stacked)
        v6_model, v7_model = models(kind)
        for (method_v6, method_v7, label) in ((ReactiveMP.Linearization(), A.Linearization(), "Linearization"), (ReactiveMP.Unscented(2), A.Unscented(2), "Unscented"))
            meta, algorithm = ReactiveMP.FlowMeta(v6_model, method_v6), F.FlowApproximation(v7_model; method = method_v7)
            inputs = (
                MvNormalMeanCovariance([0.5, -0.3], [0.6 0.1; 0.1 0.4]),
                MvNormalMeanPrecision([0.5, -0.3], [2.0 0.3; 0.3 1.5]),
                MvNormalWeightedMeanPrecision([0.8, -0.2], [2.0 0.3; 0.3 1.5]),
            )
            for input in inputs, target in (:out, :in)
                other = target === :out ? :in : :out
                m = NamedTuple{(other,)}((input,))
                v7 = call_message_update_rule(F.Flow, target; m, algorithm)
                v6, _ = v6_message_update(ReactiveMP.Flow, target, m, NamedTuple(); meta)
                id = "Flow:$target:$kind:$label:$(nameof(typeof(input)))"
                @test compare_with_reference(id, v7, v6; node = "Flow", target = ":$target", atol = 1.0e-8).outcome === :agree
            end
            # v6's marginal over `(in,)`: the marginal of `in`, m_in times the message towards `in`.
            m_out, m_in = inputs[1], MvNormalMeanCovariance([0.1, 0.2], [1.0 0.0; 0.0 1.0])
            v6 = v6_marginal_update(ReactiveMP.Flow, (:in,), (out = m_out, in = m_in), NamedTuple(); meta)
            v7 = prod(GenericProd(), m_in, call_message_update_rule(F.Flow, :in; m = (out = m_out,), algorithm))
            @test isapprox(mean(v7), mean(v6); atol = 1.0e-8) && isapprox(cov(v7), cov(v6); atol = 1.0e-8)
        end
    end
end
