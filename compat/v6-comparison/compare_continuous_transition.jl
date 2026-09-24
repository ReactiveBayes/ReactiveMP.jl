# ContinuousTransitionMessagePassingRules compared with v6's ContinuousTransition on identical
# inputs: a linear f (reshape, with dx = dy and dx ≠ dy both ways), a rotation and an affine f.
# Every rule, mean-field and structured, the joint, and both energies. Two corrections are
# declared: v6's energies, wrong in three terms for every f, and its rules towards `a` and `W`,
# which dropped the offset of an affine or nonlinear f.
#
#   julia --startup-file=no --project=compat/v6-comparison compat/v6-comparison/compare_continuous_transition.jl

include(joinpath(@__DIR__, "V6Oracle.jl"))

using .V6Oracle, Test
using ExponentialFamily, BayesBase, Distributions, LinearAlgebra
using MessagePassingRulesBase, MessagePassingRulesTestUtils, ContinuousTransitionMessagePassingRules
import ReactiveMP

const ENERGY = """
v6's average energies took ⟨log det W⟩ without the half, the dimension of the log 2π term as half
of q(y)'s (mean-field) or of q(y, x)'s (structured), and the identity for Vx beside the
uncertainty of `a`. The port computes the closed form, which agrees with a Monte Carlo estimate;
the package's energy tests pin it.
"""

const OFFSET = """
v6's rules towards `a` and `W` took each row of A = f(a) as linear in `a` through the origin,
dropping the offset f(m_a) - J m_a of an affine or nonlinear f that the other rules and the energy
keep. The port uses the same linearisation in every rule; the package's tests pin an affine f
against the linear model on y - B x (user, 2026-09-24).
"""

spd(d, s) = [i == j ? s + i : 0.2 / (i + j) for i in 1:d, j in 1:d]
spread(a, b, d) = [a + (b - a) * (k - 1) / max(d - 1, 1) for k in 1:d]

rotation(a) = [cos(a[1]) -sin(a[1]); sin(a[1]) cos(a[1])]
const B = [0.5 -0.2; 0.1 0.3]

# (name, f, dy, dx, length of a, whether f is linear through the origin)
const TRANSFORMATIONS = [
    ("reshape 2×2", a -> reshape(a, 2, 2), 2, 2, 4, true),
    ("reshape 1×2", a -> reshape(a, 1, 2), 1, 2, 2, true),
    ("reshape 2×3", a -> reshape(a, 2, 3), 2, 3, 6, true),
    ("reshape 3×2", a -> reshape(a, 3, 2), 3, 2, 6, true),
    ("rotation", rotation, 2, 2, 1, false),
    ("affine", a -> B + reshape(a, 2, 2), 2, 2, 4, false),
]

@testset "ContinuousTransitionMessagePassingRules against v6" begin
    for (name, f, dy, dx, da, linear) in TRANSFORMATIONS
        algorithm, meta = CTVMP(f), ReactiveMP.CTMeta(f)
        q_a = MvNormalMeanCovariance(spread(0.3, -0.4, da), 0.05 * spd(da, 1.0))
        q_W = Wishart(dy + 3, spd(dy, 0.5) / 3)
        q_y, q_x = MvNormalMeanCovariance(spread(1.0, 0.2, dy), 0.3 * spd(dy, 1.0)), MvNormalMeanCovariance(spread(-0.5, 0.8, dx), 0.4 * spd(dx, 1.0))
        m_y, m_x = MvNormalMeanCovariance(spread(1.1, 0.3, dy), spd(dy, 1.0)), MvNormalMeanCovariance(spread(-0.7, 0.6, dx), spd(dx, 2.0))
        q_y_x = MvNormalMeanCovariance(vcat(mean(q_y), mean(q_x)), 0.3 * spd(dy + dx, 1.5))

        cases = [
            (:y, "structured", (x = m_x,), (a = q_a, W = q_W), ()),
            (:y, "mean-field", NamedTuple(), (x = q_x, a = q_a, W = q_W), ()),
            (:x, "structured", (y = m_y,), (a = q_a, W = q_W), ()),
            (:x, "mean-field", NamedTuple(), (y = q_y, a = q_a, W = q_W), ()),
            (:a, "structured", NamedTuple(), (a = q_a, W = q_W), ((:y, :x) => q_y_x,)),
            (:a, "mean-field", NamedTuple(), (y = q_y, x = q_x, a = q_a, W = q_W), ()),
            (:W, "structured", NamedTuple(), (a = q_a,), ((:y, :x) => q_y_x,)),
            (:W, "mean-field", NamedTuple(), (y = q_y, x = q_x, a = q_a), ()),
        ]
        for (target, form, m, q, clusters) in cases
            v7 = call_message_update_rule(ContinuousTransition, target; m, q, clusters, algorithm)
            v6_q = isempty(clusters) ? q : merge((y_x = last(only(clusters)),), q)
            v6, _ = v6_message_update(ReactiveMP.ContinuousTransition, target, m, v6_q; meta)
            id = "ContinuousTransition:$target:$form:$name"
            corrected = !linear && target in (:a, :W)
            declared = corrected ? [DeclaredDisagreement(id; kind = :correction, reasoning = OFFSET)] : DeclaredDisagreement[]
            @test compare_with_reference(id, v7, v6; node = "ContinuousTransition", target = ":$target", declared).outcome === (corrected ? :correction : :agree)
        end

        v7 = call_marginal_update_rule(ContinuousTransition, (:y, :x); m = (y = m_y, x = m_x), q = (a = q_a, W = q_W), algorithm)
        v6 = v6_marginal_update(ReactiveMP.ContinuousTransition, (:y, :x), (y = m_y, x = m_x), (a = q_a, W = q_W); meta)
        @test compare_with_reference("ContinuousTransition:joint:$name", v7, v6; node = "ContinuousTransition", target = "(:y, :x)").outcome === :agree

        for (form, v7, v6) in (
                ("structured", call_average_energy(ContinuousTransition; clusters = ((:y, :x) => q_y_x,), q = (a = q_a, W = q_W), algorithm), v6_average_energy(ReactiveMP.ContinuousTransition, (a = q_a, W = q_W), ((:y, :x) => q_y_x,); meta)),
                ("mean-field", call_average_energy(ContinuousTransition; q = (y = q_y, x = q_x, a = q_a, W = q_W), algorithm), v6_average_energy(ReactiveMP.ContinuousTransition, (y = q_y, x = q_x, a = q_a, W = q_W); meta)),
            )
            id = "ContinuousTransition:energy:$form:$name"
            declared = [DeclaredDisagreement(id; kind = :correction, reasoning = ENERGY)]
            @test compare_with_reference(id, v7, v6; node = "ContinuousTransition", target = "energy", declared).outcome === :correction
        end
    end
end
