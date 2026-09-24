# AutoregressiveMessagePassingRules compared with v6's AR and ConjugateAR on identical inputs: a
# univariate AR(1) and multivariate ARs of orders 2 and 3, under ARsafe and ARunsafe. The
# messages under mean-field and the structured q(y, x), the joint and the energies. v6's
# ARunsafe joint was wrong, and is declared a correction; ConjugateAR's single-interface `w`
# marginal is not ported.
#
#   julia --startup-file=no --project=compat/v6-comparison compat/v6-comparison/compare_autoregressive.jl

include(joinpath(@__DIR__, "V6Oracle.jl"))

using .V6Oracle, Test
using ExponentialFamily, BayesBase, Distributions, LinearAlgebra
using MessagePassingRulesBase, MessagePassingRulesTestUtils, AutoregressiveMessagePassingRules
import ReactiveMP

const ARUNSAFE_JOINT = """
v6's ARunsafe joint q(y, x) inverted its blocks with a Schur complement of the wrong sign, and so
disagreed with ARsafe even for a univariate AR(1), where ARsafe regularises nothing; for a
multivariate AR it inverted the companion matrix, which is singular, and threw. The port
computes it by the Kalman gain, exactly; the AR package's marginal tests pin it against the
closed-form joint.
"""

# A normal over d dimensions, univariate when the AR is.
normal(form, m, v) = form === Univariate ? NormalMeanVariance(only(m), only(v)) : MvNormalMeanCovariance(m, v)
spd(d, s) = [i == j ? s + i : 0.2 / (i + j) for i in 1:d, j in 1:d]
spread(a, b, d) = [a + (b - a) * (k - 1) / max(d - 1, 1) for k in 1:d]

# A message towards `y` from q(x) has the lazy noise covariance, each package's own
# ARTransitionMatrix type; the comparison is of its entries.
dense(d::MvNormalMeanCovariance) = MvNormalMeanCovariance(mean(d), Matrix(cov(d)))
dense(d) = d

# v6's multivariate ARunsafe joint throws; the exception is what it gives.
attempt(f) =
try
    f()
catch e
    e
end

v6_stype(::ARsafe) = ReactiveMP.ARsafe()
v6_stype(::ARunsafe) = ReactiveMP.ARunsafe()

@testset "AutoregressiveMessagePassingRules against v6" begin
    for (form, order) in ((Univariate, 1), (Multivariate, 2), (Multivariate, 3)), stype in (ARsafe(), ARunsafe())
        algorithm = ARVMP(form, order, stype)
        meta = ReactiveMP.ARMeta(form, order, v6_stype(stype))
        label = "$(form):order $order:$(nameof(typeof(stype)))"

        q_θ = normal(form, spread(0.5, -0.3, order), 0.1 * spd(order, 1.0))
        q_γ = GammaShapeRate(3.0, 2.0)
        q_y = normal(form, spread(1.2, 0.2, order), spd(order, 0.7))
        q_x = normal(form, spread(-1.0, 2.0, order), spd(order, 0.5))
        m_y = normal(form, spread(-0.4, 0.6, order), spd(order, 1.5))
        m_x = normal(form, spread(0.3, 1.1, order), spd(order, 2.0))
        joint = MvNormalMeanCovariance(spread(0.4, 1.3, 2order), [i == j ? 1.0 + i : 0.1 for i in 1:2order, j in 1:2order])

        # AR, every message.
        cases = [
            (:y, (x = m_x,), (θ = q_θ, γ = q_γ), ()),
            (:y, NamedTuple(), (x = q_x, θ = q_θ, γ = q_γ), ()),
            (:x, (y = m_y,), (θ = q_θ, γ = q_γ), ()),
            (:x, NamedTuple(), (y = q_y, θ = q_θ, γ = q_γ), ()),
            (:θ, NamedTuple(), (γ = q_γ,), ((:y, :x) => joint,)),
            (:θ, NamedTuple(), (y = q_y, x = q_x, γ = q_γ), ()),
            (:γ, NamedTuple(), (θ = q_θ,), ((:y, :x) => joint,)),
            (:γ, NamedTuple(), (y = q_y, x = q_x, θ = q_θ), ()),
        ]
        for (target, m, q, clusters) in cases
            v7 = call_message_update_rule(AR, target; m, q, clusters, algorithm)
            v6_q = isempty(clusters) ? q : merge((y_x = last(only(clusters)),), q)
            v6, _ = v6_message_update(ReactiveMP.AR, target, m, v6_q; meta)
            @test compare_with_reference("AR:$target:$label", dense(v7), dense(v6); node = "AR", target = ":$target").outcome === :agree
        end

        # AR, the joint: ARsafe agrees, ARunsafe is v6's bug corrected.
        v7 = call_marginal_update_rule(AR, (:y, :x); m = (y = m_y, x = m_x), q = (θ = q_θ, γ = q_γ), algorithm)
        v6 = attempt(() -> v6_marginal_update(ReactiveMP.AR, (:y, :x), (y = m_y, x = m_x), (θ = q_θ, γ = q_γ); meta))
        id = "AR:joint:$label"
        declared = stype === ARunsafe() ? [DeclaredDisagreement(id; kind = :correction, reasoning = ARUNSAFE_JOINT)] : DeclaredDisagreement[]
        @test compare_with_reference(id, v7, v6; node = "AR", target = "(:y, :x)", declared).outcome === (stype === ARunsafe() ? :correction : :agree)

        # AR, the energies.
        v7 = call_average_energy(AR; clusters = ((:y, :x) => joint,), q = (θ = q_θ, γ = q_γ), algorithm)
        v6 = v6_average_energy(ReactiveMP.AR, (θ = q_θ, γ = q_γ), ((:y, :x) => joint,); meta)
        @test compare_with_reference("AR:energy:structured:$label", v7, v6; node = "AR", target = "energy").outcome === :agree
        v7 = call_average_energy(AR; q = (y = q_y, x = q_x, θ = q_θ, γ = q_γ), algorithm)
        v6 = v6_average_energy(ReactiveMP.AR, (y = q_y, x = q_x, θ = q_θ, γ = q_γ); meta)
        @test compare_with_reference("AR:energy:meanfield:$label", v7, v6; node = "AR", target = "energy").outcome === :agree

        # ConjugateAR, under the structured q(y, x) q(w) its rules take, for a multivariate AR.
        form === Multivariate || continue
        q_w = MvNormalGamma(spread(0.5, -0.3, order), 10.0 * spd(order, 1.0), 3.0, 2.0)
        cases = [
            (:y, (x = m_x,), (w = q_w,), ()),
            (:y, NamedTuple(), (x = q_x, w = q_w), ()),
            (:x, (y = m_y,), (w = q_w,), ()),
            (:x, NamedTuple(), (y = q_y, w = q_w), ()),
            (:w, NamedTuple(), NamedTuple(), ((:y, :x) => joint,)),
        ]
        for (target, m, q, clusters) in cases
            v7 = call_message_update_rule(ConjugateAR, target; m, q, clusters, algorithm)
            v6_q = isempty(clusters) ? q : merge((y_x = last(only(clusters)),), q)
            v6, _ = v6_message_update(ReactiveMP.ConjugateAR, target, m, v6_q; meta)
            @test compare_with_reference("ConjugateAR:$target:$label", dense(v7), dense(v6); node = "ConjugateAR", target = ":$target").outcome === :agree
        end
        v7 = call_marginal_update_rule(ConjugateAR, (:y, :x); m = (y = m_y, x = m_x), q = (w = q_w,), algorithm)
        v6 = attempt(() -> v6_marginal_update(ReactiveMP.ConjugateAR, (:y, :x), (y = m_y, x = m_x), (w = q_w,); meta))
        id = "ConjugateAR:joint:$label"
        declared = stype === ARunsafe() ? [DeclaredDisagreement(id; kind = :correction, reasoning = ARUNSAFE_JOINT)] : DeclaredDisagreement[]
        @test compare_with_reference(id, v7, v6; node = "ConjugateAR", target = "(:y, :x)", declared).outcome === (stype === ARunsafe() ? :correction : :agree)
        v7 = call_average_energy(ConjugateAR; clusters = ((:y, :x) => joint,), q = (w = q_w,), algorithm)
        v6 = v6_average_energy(ReactiveMP.ConjugateAR, (w = q_w,), ((:y, :x) => joint,); meta)
        @test compare_with_reference("ConjugateAR:energy:$label", v7, v6; node = "ConjugateAR", target = "energy").outcome === :agree
    end
end
