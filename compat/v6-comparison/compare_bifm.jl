# BIFMMessagePassingRules compared with v6's BIFM and BIFMHelper on identical inputs. v6's BIFM
# rules read a cache its rule towards `zprev` filled and its rule towards `znext` completed; the
# port's are stateless, reading their own edge's message instead. So each v6 rule runs on a fresh
# meta after v6's `zprev` and `znext` rules on the same messages, as a pass through the chain
# leaves it, and the port must agree call by call.
#
#   julia --startup-file=no --project=compat/v6-comparison compat/v6-comparison/compare_bifm.jl

include(joinpath(@__DIR__, "V6Oracle.jl"))

using .V6Oracle, Test
using ExponentialFamily, BayesBase, Distributions, LinearAlgebra
using MessagePassingRulesBase, MessagePassingRulesTestUtils, BIFMMessagePassingRules
using BayesBase: TerminalProdArgument
import ReactiveMP

spd(d, s) = [i == j ? s + i : 0.2 / (i + j) for i in 1:d, j in 1:d]
spread(a, b, d) = [a + (b - a) * (k - 1) / max(d - 1, 1) for k in 1:d]

# (name, A, B, C): a square B, a narrow one, and a scalar output.
const SLICES = [
    ("2 states, 2 inputs, 2 outputs", [0.9 0.1; 0.0 0.8], [1.0 0.2; 0.5 1.0], [1.0 0.0; 0.3 1.0]),
    ("3 states, 1 input, 2 outputs", [0.9 0.1 0.0; 0.0 0.8 0.1; 0.1 0.0 0.7], reshape([1.0, 0.5, -0.2], 3, 1), [1.0 0.0 0.2; 0.0 1.0 0.0]),
    ("2 states, 1 input, 1 output", [0.9 0.1; 0.0 0.8], reshape([1.0, 0.5], 2, 1), [1.0 0.0]),
]

@testset "BIFMMessagePassingRules against v6" begin
    for (name, A, B, C) in SLICES
        dz, du, dy = size(A, 1), size(B, 2), size(C, 1)
        m_out = MvNormalMeanPrecision(spread(1.0, 0.5, dy), spd(dy, 1.5))
        m_in = MvNormalMeanPrecision(spread(0.2, -0.1, du), spd(du, 0.8))
        m_znext = MvNormalWeightedMeanPrecision(spread(0.3, 0.1, dz), spd(dz, 0.4))
        m_zprev = TerminalProdArgument(MvNormalMeanCovariance(spread(0.4, -0.2, dz), spd(dz, 0.6)))
        algorithm = BIFMSmoother(A, B, C)
        all = (out = m_out, in = m_in, zprev = m_zprev, znext = m_znext)

        # v6 on a meta the backward and forward passes have filled.
        function v6_after_pass(target)
            meta = ReactiveMP.BIFMMeta(A, B, C)
            v6_message_update(ReactiveMP.BIFM, :zprev, (out = m_out, in = m_in, znext = m_znext), NamedTuple(); meta)
            target === :zprev && return first(v6_message_update(ReactiveMP.BIFM, :zprev, (out = m_out, in = m_in, znext = m_znext), NamedTuple(); meta))
            v6_message_update(ReactiveMP.BIFM, :znext, (out = m_out, in = m_in, zprev = m_zprev), NamedTuple(); meta)
            inputs = Base.structdiff(all, NamedTuple{(target,)})
            return first(v6_message_update(ReactiveMP.BIFM, target, inputs, NamedTuple(); meta))
        end
        v7_inputs(target) = target === :zprev ? (out = m_out, in = m_in, znext = m_znext) : all
        for target in (:zprev, :znext, :in, :out)
            v7 = getresult(call_message_update_rule(BIFM, target; m = v7_inputs(target), algorithm))
            @test compare_with_reference("BIFM:$target:$name", v7, v6_after_pass(target); node = "BIFM", target = ":$target").outcome === :agree
        end
    end

    # BIFMHelper passes the backward message through, and starts the forward pass from q(in).
    m_out = MvNormalWeightedMeanPrecision([0.3, 0.1], [0.5 0.1; 0.1 0.4])
    q_in = MvNormalMeanCovariance([0.4, -0.2], [0.6 0.1; 0.1 0.7])
    v6, _ = v6_message_update(ReactiveMP.BIFMHelper, :in, (out = m_out,), NamedTuple())
    @test compare_with_reference("BIFMHelper:in", getresult(call_message_update_rule(BIFMHelper, :in; m = (out = m_out,))), v6; node = "BIFMHelper", target = ":in").outcome === :agree
    v6, _ = v6_message_update(ReactiveMP.BIFMHelper, :out, NamedTuple(), (in = q_in,))
    @test compare_with_reference("BIFMHelper:out", getresult(call_message_update_rule(BIFMHelper, :out; q = (in = q_in,))), v6; node = "BIFMHelper", target = ":out").outcome === :agree
end
