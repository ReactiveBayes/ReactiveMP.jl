# The rules ported into StandardMessagePassingRules, compared with their v6 originals on
# identical inputs. A difference fails unless it is declared, with its reasoning.
#
#   julia --startup-file=no --project=compat/v6-comparison compat/v6-comparison/compare_standard.jl

include(joinpath(@__DIR__, "V6Oracle.jl"))

using .V6Oracle, Test
using ExponentialFamily, BayesBase, Distributions
using MessagePassingRulesBase, MessagePassingRulesTestUtils, StandardMessagePassingRules
using MessagePassingRulesBase: AnnotationStore, getannotation

# ReactiveMP.jl#669: v6's variational NormalMeanVariance rules take the variance a `q_v`
# contributes as E[v]; naive VMP gives 1/E[1/v], which the ports use. They agree for a point
# mass, and the ports verify against the node definition (StandardMessagePassingRules tests).
const NMV_669 = "ReactiveMP.jl#669: v6 uses E[v] for the variance a non-point-mass q_v contributes; naive VMP gives 1/E[1/v], and the port verifies against the node definition"

const INVERSE_GAMMA = InverseGamma(3.0, 4.0)

# (id, node, target, inputs, declared?)
const MESSAGE_CASES = [
    ("NMV:out:m-point-masses", NormalMeanVariance, :out, (m = (μ = PointMass(1.0), v = PointMass(2.0)),), false),
    ("NMV:out:m-normal", NormalMeanVariance, :out, (m = (μ = NormalMeanVariance(0.5, 1.5), v = PointMass(2.0)),), false),
    ("NMV:out:m-precision", NormalMeanVariance, :out, (m = (μ = NormalMeanPrecision(0.5, 4.0), v = PointMass(0.25)),), false),
    ("NMV:out:q-point-masses", NormalMeanVariance, :out, (q = (μ = PointMass(1.0), v = PointMass(2.0)),), false),
    ("NMV:out:q-normal-point-mass", NormalMeanVariance, :out, (q = (μ = NormalMeanVariance(1.0, 2.0), v = PointMass(2.0)),), false),
    ("NMV:out:q-inverse-gamma", NormalMeanVariance, :out, (q = (μ = NormalMeanVariance(1.0, 2.0), v = INVERSE_GAMMA),), true),
    ("NMV:out:m-point-mass-q-inverse-gamma", NormalMeanVariance, :out, (m = (μ = PointMass(-1.0),), q = (v = INVERSE_GAMMA,)), true),
    ("NMV:out:m-normal-q-inverse-gamma", NormalMeanVariance, :out, (m = (μ = NormalMeanVariance(0.0, 1.0),), q = (v = INVERSE_GAMMA,)), true),
    ("NMV:out:m-normal-q-point-mass", NormalMeanVariance, :out, (m = (μ = NormalWeightedMeanPrecision(2.0, 0.5),), q = (v = PointMass(1.0),)), false),
    ("NMV:μ:m-point-masses", NormalMeanVariance, :μ, (m = (out = PointMass(1.0), v = PointMass(2.0)),), false),
    ("NMV:μ:m-normal", NormalMeanVariance, :μ, (m = (out = NormalMeanVariance(-1.0, 0.5), v = PointMass(3.0)),), false),
    ("NMV:μ:q-point-masses", NormalMeanVariance, :μ, (q = (out = PointMass(1.0), v = PointMass(2.0)),), false),
    ("NMV:μ:q-inverse-gamma", NormalMeanVariance, :μ, (q = (out = NormalMeanVariance(1.0, 2.0), v = INVERSE_GAMMA),), true),
    ("NMV:μ:m-point-mass-q-inverse-gamma", NormalMeanVariance, :μ, (m = (out = PointMass(-1.0),), q = (v = INVERSE_GAMMA,)), true),
    ("NMV:μ:m-normal-q-point-mass", NormalMeanVariance, :μ, (m = (out = NormalMeanVariance(0.0, 1.0),), q = (v = PointMass(1.5),)), false),
    ("NMV:μ:m-normal-q-inverse-gamma", NormalMeanVariance, :μ, (m = (out = NormalMeanVariance(0.0, 1.0),), q = (v = INVERSE_GAMMA,)), true),
    ("NMP:out:m-point-masses", NormalMeanPrecision, :out, (m = (μ = PointMass(1.0), τ = PointMass(2.0)),), false),
    ("NMP:out:m-normal", NormalMeanPrecision, :out, (m = (μ = NormalMeanVariance(0.0, 1.0), τ = PointMass(2.0)),), false),
    ("NMP:out:q-point-masses", NormalMeanPrecision, :out, (q = (μ = PointMass(1.0), τ = PointMass(2.0)),), false),
    ("NMP:out:q-gamma", NormalMeanPrecision, :out, (q = (μ = NormalMeanVariance(1.0, 2.0), τ = GammaShapeRate(3.0, 2.0)),), false),
    ("NMP:out:m-point-mass-q-gamma", NormalMeanPrecision, :out, (m = (μ = PointMass(-1.0),), q = (τ = GammaShapeRate(3.0, 2.0),)), false),
    ("NMP:out:m-normal-q-gamma", NormalMeanPrecision, :out, (m = (μ = NormalWeightedMeanPrecision(1.0, 2.0),), q = (τ = GammaShapeRate(3.0, 2.0),)), false),
    ("NMP:μ:m-point-masses", NormalMeanPrecision, :μ, (m = (out = PointMass(1.0), τ = PointMass(2.0)),), false),
    ("NMP:μ:m-normal", NormalMeanPrecision, :μ, (m = (out = NormalMeanVariance(0.0, 1.0), τ = PointMass(2.0)),), false),
    ("NMP:μ:q-point-masses", NormalMeanPrecision, :μ, (q = (out = PointMass(1.0), τ = PointMass(2.0)),), false),
    ("NMP:μ:q-gamma", NormalMeanPrecision, :μ, (q = (out = PointMass(1.2), τ = GammaShapeRate(3.0, 2.0)),), false),
    ("NMP:μ:m-normal-q-gamma", NormalMeanPrecision, :μ, (m = (out = NormalMeanVariance(0.0, 1.0),), q = (τ = GammaShapeRate(3.0, 2.0),)), false),
    ("NMP:τ:q-normals", NormalMeanPrecision, :τ, (q = (out = PointMass(1.2), μ = NormalWeightedMeanPrecision(1.0, 2.0)),), false),
    ("GSR:out:m-point-masses", GammaShapeRate, :out, (m = (α = PointMass(2.0), β = PointMass(3.0)),), false),
    ("GSR:out:q", GammaShapeRate, :out, (q = (α = PointMass(2.0), β = GammaShapeRate(2.0, 4.0)),), false),
    ("Categorical:out:m-dirichlet", Categorical, :out, (m = (p = Dirichlet([1.0, 3.0]),),), false),
    ("Categorical:out:q-dirichlet", Categorical, :out, (q = (p = Dirichlet([1.0, 3.0, 0.5]),),), false),
    ("Categorical:out:m-point-mass", Categorical, :out, (m = (p = PointMass([0.2, 0.8]),),), false),
    ("Categorical:out:q-point-mass", Categorical, :out, (q = (p = PointMass([0.2, 0.8]),),), false),
    ("Categorical:p:q-categorical", Categorical, :p, (q = (out = Categorical([0.3, 0.2, 0.5]),),), false),
    ("Categorical:p:q-one-hot", Categorical, :p, (q = (out = PointMass([0.0, 1.0, 0.0]),),), false),
    ("Dirichlet:out:m", Dirichlet, :out, (m = (a = PointMass([1.0, 2.0]),),), false),
    ("Dirichlet:out:q", Dirichlet, :out, (q = (a = PointMass([1.0, 2.0, 4.0]),),), false),
]

const CLUSTER_MESSAGE_CASES = [
    ("NMP:τ:q-joint", NormalMeanPrecision, :τ, ((:out, :μ) => MvNormalWeightedMeanPrecision([1.0, 0.5], [3.0 -1.0; -1.0 2.0]),), false),
]

const MARGINAL_CASES = [
    ("NMV:(out,μ):point-mass-v", NormalMeanVariance, (:out, :μ), (m = (out = NormalMeanVariance(1.0, 1.0), μ = NormalMeanVariance(2.0, 0.5)), q = (v = PointMass(2.0),)), false),
    ("NMV:(out,μ):inverse-gamma-v", NormalMeanVariance, (:out, :μ), (m = (out = NormalMeanVariance(1.0, 1.0), μ = NormalMeanVariance(2.0, 0.5)), q = (v = INVERSE_GAMMA,)), true),
    ("NMP:(out,μ):gamma-τ", NormalMeanPrecision, (:out, :μ), (m = (out = NormalMeanVariance(1.0, 1.0), μ = NormalWeightedMeanPrecision(2.0, 0.5)), q = (τ = GammaShapeRate(3.0, 2.0),)), false),
]

const AVERAGE_ENERGY_CASES = [
    ("NMV:energy:singles", NormalMeanVariance, (q = (out = NormalMeanVariance(0.0, 1.0), μ = NormalMeanVariance(1.0, 2.0), v = INVERSE_GAMMA),), false),
    ("NMV:energy:joint", NormalMeanVariance, (q = (v = PointMass(2.0),), clusters = ((:out, :μ) => MvNormalMeanCovariance([0.0, 1.0], [1.0 0.2; 0.2 2.0]),)), false),
    ("NMP:energy:singles", NormalMeanPrecision, (q = (out = PointMass(1.2), μ = NormalWeightedMeanPrecision(1.0, 2.0), τ = GammaShapeRate(3.0, 2.0)),), false),
    ("GSR:energy:point-mass-α", GammaShapeRate, (q = (out = GammaShapeRate(2.0, 3.0), α = PointMass(2.0), β = GammaShapeRate(3.0, 1.5)),), false),
    ("GSR:energy:gamma-α", GammaShapeRate, (q = (out = GammaShapeRate(2.0, 3.0), α = GammaShapeRate(4.0, 2.0), β = PointMass(3.0)),), false),
    ("Categorical:energy", Categorical, (q = (out = Categorical([0.3, 0.2, 0.5]), p = Dirichlet([1.0, 3.0, 0.5])),), false),
    ("Dirichlet:energy", Dirichlet, (q = (out = Dirichlet([2.0, 3.0]), a = PointMass([2.0, 1.0])),), false),
    ("NMP:energy:joint", NormalMeanPrecision, (q = (τ = GammaShapeRate(3.0, 2.0),), clusters = ((:out, :μ) => MvNormalWeightedMeanPrecision([1.0, 0.5], [3.0 -1.0; -1.0 2.0]),)), false),
]

# NormalMixture: v7's node is the new package's own type, v6's is `NormalMixture{N}`. v6 gives
# an indexed target the aligned member alone (`q_p` for `(:m, k)`), where v7 passes the
# group in full with `nothing` for the members the dependency leaves out.
const V6_NORMAL_MIXTURE = V6Oracle.ReactiveMP.NormalMixture{2}
aligned(group, k) = group[k]

const MIXTURE_COMPONENTS = (m = (NormalMeanVariance(0.0, 1.0), NormalWeightedMeanPrecision(2.0, 1.5)), p = (GammaShapeRate(2.0, 1.0), GammaShapeRate(3.0, 2.0)))
const MIXTURE_SWITCH = Categorical([0.3, 0.7])

const MIXTURE_CASES = [
    ("NormalMixture:(m,1)", (:m, 1), (q = (out = PointMass(1.5), switch = MIXTURE_SWITCH, p = (MIXTURE_COMPONENTS.p[1], nothing)),), (q = (out = PointMass(1.5), switch = MIXTURE_SWITCH, p = MIXTURE_COMPONENTS.p[1]),)),
    ("NormalMixture:(m,2)", (:m, 2), (q = (out = NormalMeanVariance(1.0, 0.5), switch = MIXTURE_SWITCH, p = (nothing, MIXTURE_COMPONENTS.p[2])),), (q = (out = NormalMeanVariance(1.0, 0.5), switch = MIXTURE_SWITCH, p = MIXTURE_COMPONENTS.p[2]),)),
    ("NormalMixture:(p,1)", (:p, 1), (q = (out = PointMass(1.5), switch = MIXTURE_SWITCH, m = (MIXTURE_COMPONENTS.m[1], nothing)),), (q = (out = PointMass(1.5), switch = MIXTURE_SWITCH, m = MIXTURE_COMPONENTS.m[1]),)),
    ("NormalMixture:(p,2)", (:p, 2), (q = (out = PointMass(1.5), switch = MIXTURE_SWITCH, m = (nothing, MIXTURE_COMPONENTS.m[2])),), (q = (out = PointMass(1.5), switch = MIXTURE_SWITCH, m = MIXTURE_COMPONENTS.m[2]),)),
    ("NormalMixture:switch", :switch, (q = (out = PointMass(1.5), MIXTURE_COMPONENTS...),), (q = (out = PointMass(1.5), MIXTURE_COMPONENTS...),)),
    ("NormalMixture:out", :out, (q = (switch = MIXTURE_SWITCH, MIXTURE_COMPONENTS...),), (q = (switch = MIXTURE_SWITCH, MIXTURE_COMPONENTS...),)),
]

declare(id, flagged) = flagged ? [DeclaredDisagreement(id; kind = :correction, reasoning = NMV_669)] : DeclaredDisagreement[]

@testset "StandardMessagePassingRules against v6" begin
    @testset "message rules" begin
        for (id, node, edge, inputs, flagged) in MESSAGE_CASES
            m, q = get(inputs, :m, NamedTuple()), get(inputs, :q, NamedTuple())
            store = AnnotationStore()
            v7 = call_message_update_rule(node, edge; m, q, ann = store)
            v6, v6_logscale = v6_message_update(node, edge, m, q)
            record = compare_with_reference(id, v7, v6; inputs, node = string(node), target = ":$edge", v7_logscale = getannotation(store, :logscale, nothing), v6_logscale, declared = declare(id, flagged))
            # A declared correction must actually differ, or the declaration is stale.
            @test flagged == (record.outcome === :correction)
        end
    end
    @testset "message rules consuming a joint" begin
        for (id, node, edge, clusters, flagged) in CLUSTER_MESSAGE_CASES
            v7 = call_message_update_rule(node, edge; clusters)
            v6, _ = v6_message_update(node, edge, NamedTuple(), NamedTuple{map(V6Oracle.v6_name, Tuple(first.(clusters)))}(Tuple(last.(clusters))))
            record = compare_with_reference(id, v7, v6; inputs = clusters, node = string(node), target = ":$edge", declared = declare(id, flagged))
            @test flagged == (record.outcome === :correction)
        end
    end
    @testset "NormalMixture" begin
        for (id, target, v7_inputs, v6_inputs) in MIXTURE_CASES
            v7 = call_message_update_rule(NormalMixture, target; q = v7_inputs.q)
            v6, _ = v6_message_update(V6_NORMAL_MIXTURE, target, NamedTuple(), v6_inputs.q)
            @test compare_with_reference(id, v7, v6; inputs = v7_inputs, node = "NormalMixture", target = string(target)).outcome === :agree
        end
        q = (out = PointMass(1.5), switch = MIXTURE_SWITCH, MIXTURE_COMPONENTS...)
        v7 = call_average_energy(NormalMixture; q)
        v6 = v6_average_energy(V6_NORMAL_MIXTURE, q)
        @test compare_with_reference("NormalMixture:energy", v7, v6; node = "NormalMixture", target = "energy").outcome === :agree
    end
    @testset "marginal rules" begin
        for (id, node, members, inputs, flagged) in MARGINAL_CASES
            m, q = get(inputs, :m, NamedTuple()), get(inputs, :q, NamedTuple())
            v7 = call_marginal_update_rule(node, members; m, q)
            v6 = v6_marginal_update(node, members, m, q)
            record = compare_with_reference(id, v7, v6; inputs, node = string(node), target = string(members), declared = declare(id, flagged))
            @test flagged == (record.outcome === :correction)
        end
    end
    @testset "average energies" begin
        for (id, node, inputs, flagged) in AVERAGE_ENERGY_CASES
            q, clusters = get(inputs, :q, NamedTuple()), get(inputs, :clusters, ())
            v7 = call_average_energy(node; q, clusters)
            v6 = v6_average_energy(node, q, clusters)
            record = compare_with_reference(id, v7, v6; inputs, node = string(node), target = "energy", declared = declare(id, flagged))
            @test flagged == (record.outcome === :correction)
        end
    end
end
