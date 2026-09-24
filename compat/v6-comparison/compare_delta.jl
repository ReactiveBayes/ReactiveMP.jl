# The Delta node's rules in DeltaMessagePassingRules, compared with v6's on identical inputs,
# for both Gaussian methods. v6's rules read the function from a `DeltaFnNode`, which the oracle
# builds as v6's `@call_rule` does; the port reads it from the context's node.
#
#   julia --startup-file=no --project=compat/v6-comparison compat/v6-comparison/compare_delta.jl

include(joinpath(@__DIR__, "V6Oracle.jl"))

using .V6Oracle, Test
using ExponentialFamily, BayesBase, Distributions
using MessagePassingRulesBase, MessagePassingRulesTestUtils, DeltaMessagePassingRules
using MessagePassingRulesBase: RuleContext, Target
import MessagePassingRulesApproximations as Approximations
import ReactiveMP

# The engine's node, as far as a rule sees it: it gives the function towards `out`.
struct FunctionNode{F}
    f::F
end
MessagePassingRulesBase.getnodefn(node::FunctionNode, ::Target{:out}) = node.f

g(x) = x .^ 2 .- 5
g_inv(y) = sqrt.(y .+ 5)
h(x, y) = x .^ 2 .- y
h_inv_x(z, y) = sqrt.(z .+ y)
h_inv_z(x, y) = x .^ 2 .- y
s(x) = sin.(x) .+ x .^ 3

const METHODS = [
    ("Linearization", Approximations.Linearization(), ReactiveMP.Linearization()),
    ("Unscented", Approximations.Unscented(), ReactiveMP.Unscented()),
    ("Unscented α = 0.5", Approximations.Unscented(; alpha = 0.5, beta = 1.5, kappa = 1.0), ReactiveMP.Unscented(; alpha = 0.5, beta = 1.5, kappa = 1.0)),
]

# Towards `out`: the function and the messages from the inputs.
const OUT_CASES = [
    (g, (NormalMeanVariance(2.0, 3.0),)),
    (g, (MvNormalMeanCovariance([2.0], [3.0;;]),)),
    (s, (NormalMeanPrecision(0.3, 4.0),)),
    (h, (NormalMeanVariance(2.0, 3.0), NormalMeanVariance(5.0, 1.0))),
    (h, (MvNormalMeanCovariance([2.0], [3.0;;]), MvNormalMeanCovariance([5.0], [1.0;;]))),
    (x -> [x[1] * x[2], x[1] + x[2]], (MvNormalMeanCovariance([1.0, 2.0], [2.0 0.3; 0.3 1.0]),)),
]

# Towards an input with a known inverse: the inverse, `k`, the message from `out`, the others.
const KNOWN_INVERSE_CASES = [
    (g, g_inv, 1, NormalMeanVariance(2.0, 3.0), ()),
    (g, g_inv, 1, MvNormalMeanCovariance([2.0], [3.0;;]), ()),
    (h, (h_inv_x, h_inv_z), 1, NormalMeanVariance(2.0, 3.0), (NormalMeanVariance(5.0, 1.0),)),
    (h, (h_inv_x, h_inv_z), 2, NormalMeanVariance(2.0, 1.0), (NormalMeanVariance(5.0, 1.0),)),
]

# Towards an input without an inverse: `k`, its own message, and the joint over the inputs.
const UNKNOWN_INVERSE_CASES = [
    (1, NormalMeanVariance(5.0, 10.0), JointNormal(MvNormalMeanCovariance(ones(2), [1.0 0.1; 0.1 1.0]), ((), ()))),
    (1, MvNormalMeanCovariance([5.0], [10.0;;]), JointNormal(MvNormalMeanCovariance(ones(2), [1.0 0.1; 0.1 1.0]), ((1,), (1,)))),
    (2, NormalMeanVariance(0.0, 10.0), JointNormal(MvNormalMeanCovariance(ones(3), [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]), ((), (), ()))),
]

# The joint over the inputs: the function, the message from `out`, the inputs' messages.
const MARGINAL_CASES = [
    (g, NormalMeanVariance(2.0, 3.0), (NormalMeanVariance(2.0, 1.0),)),
    (g, MvNormalMeanCovariance([2.0], [3.0;;]), (MvNormalMeanCovariance([2.0], [1.0;;]),)),
    (h, NormalMeanVariance(2.0, 3.0), (NormalMeanVariance(2.0, 1.0), NormalMeanVariance(5.0, 1.0))),
    (h, MvNormalMeanCovariance([2.0], [3.0;;]), (MvNormalMeanCovariance([2.0], [1.0;;]), MvNormalMeanCovariance([5.0], [1.0;;]))),
]

@testset "DeltaMessagePassingRules against v6" begin
    for (label, method, v6_method) in METHODS
        algorithm, meta = DeltaApproximation(method = method), ReactiveMP.DeltaMeta(method = v6_method)
        @testset "$label towards out" begin
            for (index, (f, inputs)) in enumerate(OUT_CASES)
                v7 = call_message_update_rule(DeltaFn{typeof(f)}, :out; m = (in = inputs,), algorithm, ctx = RuleContext(node = FunctionNode(f)))
                v6, _ = v6_message_update(ReactiveMP.DeltaFn{typeof(f)}, :out, (ins = inputs,), NamedTuple(); meta, node = v6_delta_node(f, meta))
                @test compare_with_reference("Delta:$label:out:$index", v7, v6; node = "DeltaFn", target = ":out").outcome === :agree
            end
        end
        @testset "$label towards an input, known inverse" begin
            for (index, (f, inverse, k, m_out, others)) in enumerate(KNOWN_INVERSE_CASES)
                inverted = DeltaApproximation(method = method, inverse = inverse)
                inverted_meta = ReactiveMP.DeltaMeta(method = v6_method, inverse = inverse)
                group = k == 1 ? (nothing, others...) : (others..., nothing)
                v7 = call_message_update_rule(DeltaFn{typeof(f)}, (:in, k); m = (out = m_out, in = group), algorithm = inverted)
                v6_m = isempty(others) ? (out = m_out, ins = nothing) : (out = m_out, ins = others)
                v6, _ = v6_message_update(ReactiveMP.DeltaFn{typeof(f)}, (:in, k), v6_m, NamedTuple(); meta = inverted_meta, node = v6_delta_node(f, inverted_meta))
                @test compare_with_reference("Delta:$label:in:known:$index", v7, v6; node = "DeltaFn", target = "(:in, $k)").outcome === :agree
            end
        end
        @testset "$label towards an input, unknown inverse" begin
            for (index, (k, m_in, joint)) in enumerate(UNKNOWN_INVERSE_CASES)
                group = ntuple(i -> i == k ? m_in : nothing, length(joint.ds))
                v7 = call_message_update_rule(DeltaFn{typeof(h)}, (:in, k); m = (in = group,), clusters = ((:in,) => joint,), algorithm)
                v6, _ = v6_message_update(ReactiveMP.DeltaFn{typeof(h)}, (:in, k), (in = m_in,), (ins = joint,); meta, node = v6_delta_node(h, meta))
                @test compare_with_reference("Delta:$label:in:unknown:$index", v7, v6; node = "DeltaFn", target = "(:in, $k)").outcome === :agree
            end
        end
        @testset "$label joint over the inputs" begin
            for (index, (f, m_out, inputs)) in enumerate(MARGINAL_CASES)
                v7 = call_marginal_update_rule(DeltaFn{typeof(f)}, (:in,); m = (out = m_out, in = inputs), algorithm, ctx = RuleContext(node = FunctionNode(f)))
                v6 = v6_marginal_update(ReactiveMP.DeltaFn{typeof(f)}, (:ins,), (out = m_out, ins = inputs), NamedTuple(); meta, node = v6_delta_node(f, meta))
                @test compare_with_reference("Delta:$label:joint:$index", v7, v6; node = "DeltaFn", target = "(:in,)").outcome === :agree
            end
        end
    end
end
