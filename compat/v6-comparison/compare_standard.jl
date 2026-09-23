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

# v6's Gamma and GammaInverse average energies take E[x/θ] as E[x]/E[θ] and E[θ/x] as θ/E[x];
# for independent factors they are E[x]·E[1/θ] and θ·E[1/x]. They agree for point masses.
const GAMMA_ENERGY = "ReactiveMP.jl#672: v6's Gamma energy uses E[x]/E[θ] for E[x/θ]; it is E[x]·E[1/θ], and the two agree only for a point-mass q_θ"
const GAMMA_INVERSE_ENERGY = "ReactiveMP.jl#672: v6's GammaInverse energy uses θ/E[x] for E[θ/x]; it is θ·E[1/x], and the two agree only for a point-mass q_out"

const INVERSE_GAMMA = InverseGamma(3.0, 4.0)

# #669 in its multivariate form: v6's variational MvNormalMeanCovariance rules take the
# covariance a `q_Σ` contributes as E[Σ]; naive VMP gives E[Σ⁻¹]⁻¹, which its own average
# energy already uses. They agree for a point mass.
const MVNMC_669 = "ReactiveMP.jl#673, the multivariate #669: v6 uses E[Σ] for the covariance a non-point-mass q_Σ contributes; naive VMP gives E[Σ⁻¹]⁻¹, as v6's own MvNormalMeanCovariance energy does"
const I2 = [1.0 0.0; 0.0 1.0]
const INVERSE_WISHART = InverseWishart(5.0, [2.0 0.3; 0.3 2.0])
const WISHART = Wishart(4.0, [1.0 0.2; 0.2 0.5])

# v6 returns a split cluster as a NamedTuple and the port as a `FactorizedCluster`; its shape
# likelihood is v6's own type. Both are read as the port's before comparing.
as_v7(node, v6::NamedTuple) = FactorizedCluster(v6_cluster_blocks(node, v6)...)
# v6's MvNormalWeightedMeanPrecision marginal keys its blocks by its argument names, `m_out`
# and so on (ReactiveMP.jl#674); the values are right, so the keys are read without the prefix.
as_v7(::Type{MvNormalWeightedMeanPrecision}, v6::NamedTuple) =
    as_v7(nothing, NamedTuple{map(k -> Symbol(replace(string(k), r"^m_" => "")), keys(v6))}(values(v6)), MvNormalWeightedMeanPrecision)
as_v7(::Nothing, v6::NamedTuple, node) = FactorizedCluster(v6_cluster_blocks(node, v6)...)
as_v7(node, v6::V6Oracle.ReactiveMP.GammaShapeLikelihood) = GammaShapeLikelihood(params(v6)...)
as_v7(node, v6) = v6

# The nodes whose type is the port's own; v6 had its own of the same name.
const V6_NODES = Dict{Any, Any}(
    HalfNormal => V6Oracle.ReactiveMP.HalfNormal, Uninformative => V6Oracle.ReactiveMP.Uninformative,
    AND => V6Oracle.ReactiveMP.AND, OR => V6Oracle.ReactiveMP.OR, NOT => V6Oracle.ReactiveMP.NOT, IMPLY => V6Oracle.ReactiveMP.IMPLY,
)
v6_node(node) = get(V6_NODES, node, node)

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
    ("GSR:α:q", GammaShapeRate, :α, (q = (out = GammaShapeRate(2.0, 3.0), β = GammaShapeRate(3.0, 1.5)),), false),
    ("GSR:α:q-point-mass-out", GammaShapeRate, :α, (q = (out = PointMass(1.5), β = GammaShapeRate(1.0, 1.0)),), false),
    ("GSR:β:q", GammaShapeRate, :β, (q = (out = GammaShapeRate(2.0, 3.0), α = GammaShapeRate(4.0, 2.0)),), false),
    ("GSR:β:q-point-mass-α", GammaShapeRate, :β, (q = (out = GammaShapeScale(1.0, 10.0), α = PointMass(10.0)),), false),
    ("NMV:v:q-point-masses", NormalMeanVariance, :v, (q = (out = PointMass(-1.0), μ = PointMass(2.0)),), false),
    ("NMV:v:q-normals", NormalMeanVariance, :v, (q = (out = NormalMeanVariance(-1.0, 2.0), μ = NormalMeanPrecision(1.0, 4.0)),), false),
    ("Categorical:out:m-dirichlet", Categorical, :out, (m = (p = Dirichlet([1.0, 3.0]),),), false),
    ("Categorical:out:q-dirichlet", Categorical, :out, (q = (p = Dirichlet([1.0, 3.0, 0.5]),),), false),
    ("Categorical:out:m-point-mass", Categorical, :out, (m = (p = PointMass([0.2, 0.8]),),), false),
    ("Categorical:out:q-point-mass", Categorical, :out, (q = (p = PointMass([0.2, 0.8]),),), false),
    ("Categorical:p:q-categorical", Categorical, :p, (q = (out = Categorical([0.3, 0.2, 0.5]),),), false),
    ("Categorical:p:q-one-hot", Categorical, :p, (q = (out = PointMass([0.0, 1.0, 0.0]),),), false),
    ("Dirichlet:out:m", Dirichlet, :out, (m = (a = PointMass([1.0, 2.0]),),), false),
    ("Dirichlet:out:q", Dirichlet, :out, (q = (a = PointMass([1.0, 2.0, 4.0]),),), false),
    ("Beta:out:m", Beta, :out, (m = (a = PointMass(1.0), b = PointMass(2.0)),), false),
    ("Beta:out:q", Beta, :out, (q = (a = PointMass(2.5), b = PointMass(0.5)),), false),
    ("Bernoulli:out:m-beta", Bernoulli, :out, (m = (p = Beta(2.0, 6.0),),), false),
    ("Bernoulli:out:m-point-mass", Bernoulli, :out, (m = (p = PointMass(0.2),),), false),
    ("Bernoulli:out:q-point-mass", Bernoulli, :out, (q = (p = PointMass(0.3),),), false),
    ("Bernoulli:out:q-beta", Bernoulli, :out, (q = (p = Beta(1.5, 3.0),),), false),
    ("Bernoulli:p:m-point-mass", Bernoulli, :p, (m = (out = PointMass(1.0),),), false),
    ("Bernoulli:p:q-point-mass", Bernoulli, :p, (q = (out = PointMass(0.0),),), false),
    ("Bernoulli:p:q-bernoulli", Bernoulli, :p, (q = (out = Bernoulli(0.3),),), false),
    ("Bernoulli:p:q-categorical", Bernoulli, :p, (q = (out = Categorical([0.7, 0.3]),),), false),
    ("Gamma:out:m", Gamma, :out, (m = (α = PointMass(2.0), θ = PointMass(3.0)),), false),
    ("Gamma:out:q", Gamma, :out, (q = (α = GammaShapeRate(4.0, 2.0), θ = PointMass(0.5)),), false),
    ("GammaInverse:out:m", GammaInverse, :out, (m = (α = PointMass(3.0), θ = PointMass(3.0)),), false),
    ("GammaInverse:out:q", GammaInverse, :out, (q = (α = Gamma(1.0, 1.0), θ = Beta(1.0, 1.0)),), false),
    ("HalfNormal:out:q", HalfNormal, :out, (q = (v = PointMass(4.0),),), false),
    ("Poisson:out:m", Poisson, :out, (m = (l = PointMass(0.2),),), false),
    ("Poisson:out:q-gamma", Poisson, :out, (q = (l = GammaShapeRate(1.0, 0.5),),), false),
    ("Poisson:l:m", Poisson, :l, (m = (out = PointMass(3.0),),), false),
    ("Poisson:l:q", Poisson, :l, (q = (out = Poisson(0.3),),), false),
    ("Uniform:out:m-m", Uniform, :out, (m = (a = PointMass(1.0), b = PointMass(2.0)),), false),
    ("Uniform:out:q-m", Uniform, :out, (m = (b = PointMass(2.0),), q = (a = PointMass(1.0),)), false),
    ("Uniform:out:m-q", Uniform, :out, (m = (a = PointMass(1.0),), q = (b = PointMass(2.0),)), false),
    ("Uniform:out:q-q", Uniform, :out, (q = (a = PointMass(1.0), b = PointMass(2.0)),), false),
    ("MvNMC:out:m-point-masses", MvNormalMeanCovariance, :out, (m = (μ = PointMass([1.0, 3.0]), Σ = PointMass([3.0 2.0; 2.0 4.0])),), false),
    ("MvNMC:out:m-normal", MvNormalMeanCovariance, :out, (m = (μ = MvNormalMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), Σ = PointMass([6.0 4.0; 4.0 8.0])),), false),
    ("MvNMC:out:q-point-masses", MvNormalMeanCovariance, :out, (q = (μ = PointMass([1.0, 3.0]), Σ = PointMass([3.0 2.0; 2.0 4.0])),), false),
    ("MvNMC:out:q-normal-point-mass", MvNormalMeanCovariance, :out, (q = (μ = MvNormalMeanCovariance([1.0, 2.0], [3.0 2.0; 2.0 4.0]), Σ = PointMass(2 * I2)),), false),
    ("MvNMC:out:q-inverse-wishart", MvNormalMeanCovariance, :out, (q = (μ = MvNormalMeanCovariance([1.0, 2.0], [3.0 2.0; 2.0 4.0]), Σ = INVERSE_WISHART),), MVNMC_669),
    ("MvNMC:out:m-point-mass-q-inverse-wishart", MvNormalMeanCovariance, :out, (m = (μ = PointMass([1.0, 3.0]),), q = (Σ = INVERSE_WISHART,)), MVNMC_669),
    ("MvNMC:out:m-normal-q-inverse-wishart", MvNormalMeanCovariance, :out, (m = (μ = MvNormalMeanCovariance([1.0, 3.0], [1.0 0.0; 0.0 2.0]),), q = (Σ = INVERSE_WISHART,)), MVNMC_669),
    ("MvNMC:out:m-normal-q-point-mass", MvNormalMeanCovariance, :out, (m = (μ = MvNormalMeanCovariance([1.0, 3.0], [1.0 0.0; 0.0 2.0]),), q = (Σ = PointMass(2 * I2),)), false),
    ("MvNMC:μ:m-normal", MvNormalMeanCovariance, :μ, (m = (out = MvNormalMeanCovariance([0.0, 0.0], [7.0 -1.0; -1.0 9.0]), Σ = PointMass([12.0 -2.0; -2.0 7.0])),), false),
    ("MvNMC:μ:q-inverse-wishart", MvNormalMeanCovariance, :μ, (q = (out = PointMass([1.0, 2.0]), Σ = INVERSE_WISHART),), MVNMC_669),
    ("MvNMC:μ:m-point-mass-q-point-mass", MvNormalMeanCovariance, :μ, (m = (out = PointMass([1.0, 3.0]),), q = (Σ = PointMass(2 * I2),)), false),
    ("MvNMC:Σ:q", MvNormalMeanCovariance, :Σ, (q = (out = MvNormalMeanCovariance([1.0, 2.0], [3.0 2.0; 2.0 4.0]), μ = MvNormalMeanCovariance([3.0, 5.0], [3.0 2.0; 2.0 4.0])),), false),
    ("MvNMC:Σ:q-point-mass-out", MvNormalMeanCovariance, :Σ, (q = (out = PointMass([1.0, 2.0]), μ = MvNormalWeightedMeanPrecision([3.0, 5.0], [3.0 2.0; 2.0 4.0])),), false),
    ("MvNMP:out:m-point-masses", MvNormalMeanPrecision, :out, (m = (μ = PointMass([1.0, 3.0]), Λ = PointMass([3.0 2.0; 2.0 4.0])),), false),
    ("MvNMP:out:m-normal", MvNormalMeanPrecision, :out, (m = (μ = MvNormalMeanCovariance([2.0, 1.0], [3.0 2.0; 2.0 4.0]), Λ = PointMass([1.5 1.0; 1.0 2.0])),), false),
    ("MvNMP:out:q-point-masses", MvNormalMeanPrecision, :out, (q = (μ = PointMass([1.0, 3.0]), Λ = PointMass([3.0 2.0; 2.0 4.0])),), false),
    ("MvNMP:out:q-wishart", MvNormalMeanPrecision, :out, (q = (μ = MvNormalMeanCovariance([1.0, 2.0], [3.0 2.0; 2.0 4.0]), Λ = WISHART),), false),
    ("MvNMP:out:m-point-mass-q-wishart", MvNormalMeanPrecision, :out, (m = (μ = PointMass([1.0, 3.0]),), q = (Λ = WISHART,)), false),
    ("MvNMP:out:m-normal-q-wishart", MvNormalMeanPrecision, :out, (m = (μ = MvNormalMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]),), q = (Λ = WISHART,)), false),
    ("MvNMP:out:m-normal-q-point-mass", MvNormalMeanPrecision, :out, (m = (μ = MvNormalMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]),), q = (Λ = PointMass(2 * I2),)), false),
    ("MvNMP:μ:m-normal-q-wishart", MvNormalMeanPrecision, :μ, (m = (out = MvNormalMeanCovariance([0.0, 0.0], [7.0 -1.0; -1.0 9.0]),), q = (Λ = WISHART,)), false),
    ("MvNMP:μ:q-wishart", MvNormalMeanPrecision, :μ, (q = (out = PointMass([1.0, 2.0]), Λ = WISHART),), false),
    ("MvNMP:Λ:q", MvNormalMeanPrecision, :Λ, (q = (out = MvNormalMeanCovariance([1.0, 2.0], [3.0 2.0; 2.0 4.0]), μ = MvNormalMeanPrecision([3.0, 5.0], [3.0 2.0; 2.0 4.0])),), false),
    ("MvNMP:Λ:q-point-mass-out", MvNormalMeanPrecision, :Λ, (q = (out = PointMass([1.0, 2.0]), μ = MvNormalWeightedMeanPrecision([3.0, 5.0], [3.0 2.0; 2.0 4.0])),), false),
    ("MvNWMP:out:m", MvNormalWeightedMeanPrecision, :out, (m = (ξ = PointMass([1.0, 3.0]), Λ = PointMass([3.0 2.0; 2.0 4.0])),), false),
    ("MvNWMP:out:q", MvNormalWeightedMeanPrecision, :out, (q = (ξ = MvNormalMeanCovariance([1.0, 2.0], [3.0 2.0; 2.0 4.0]), Λ = WISHART),), false),
    ("AND:out", AND, :out, (m = (in1 = Bernoulli(0.3), in2 = Bernoulli(0.5)),), false),
    ("AND:in1", AND, :in1, (m = (out = Bernoulli(0.3), in2 = Bernoulli(0.4)),), false),
    ("AND:in2", AND, :in2, (m = (out = Bernoulli(0.7), in1 = Bernoulli(0.2)),), false),
    ("OR:out", OR, :out, (m = (in1 = Bernoulli(0.3), in2 = Bernoulli(0.4)),), false),
    ("OR:in1", OR, :in1, (m = (out = Bernoulli(0.3), in2 = Bernoulli(0.4)),), false),
    ("OR:in2", OR, :in2, (m = (out = Bernoulli(0.7), in1 = Bernoulli(0.2)),), false),
    ("NOT:out", NOT, :out, (m = (in = Bernoulli(0.3),),), false),
    ("NOT:in", NOT, :in, (m = (out = Bernoulli(0.6),),), false),
    ("IMPLY:out", IMPLY, :out, (m = (in1 = Bernoulli(0.4), in2 = Bernoulli(0.7)),), false),
    ("IMPLY:in1", IMPLY, :in1, (m = (out = Bernoulli(0.2), in2 = Bernoulli(0.5)),), false),
    ("IMPLY:in2", IMPLY, :in2, (m = (out = Bernoulli(0.3), in1 = Bernoulli(0.4)),), false),
]

const CLUSTER_MESSAGE_CASES = [
    ("NMP:τ:q-joint", NormalMeanPrecision, :τ, ((:out, :μ) => MvNormalWeightedMeanPrecision([1.0, 0.5], [3.0 -1.0; -1.0 2.0]),), false),
    ("MvNMC:Σ:q-joint", MvNormalMeanCovariance, :Σ, ((:out, :μ) => MvNormalMeanCovariance([1.0, 0.5, 2.0, 1.0], [2.0 0.1 0.2 0.0; 0.1 1.5 0.0 0.3; 0.2 0.0 1.0 0.1; 0.0 0.3 0.1 2.5]),), false),
    ("MvNMP:Λ:q-joint", MvNormalMeanPrecision, :Λ, ((:out, :μ) => MvNormalMeanCovariance([1.0, 0.5, 2.0, 1.0], [2.0 0.1 0.2 0.0; 0.1 1.5 0.0 0.3; 0.2 0.0 1.0 0.1; 0.0 0.3 0.1 2.5]),), false),
    ("NMV:v:q-joint", NormalMeanVariance, :v, ((:out, :μ) => MvNormalMeanCovariance([2.0, 3.0], [2.0 -0.1; -0.1 3.0]),), false),
]

const MARGINAL_CASES = [
    ("NMV:(out,μ):point-mass-v", NormalMeanVariance, (:out, :μ), (m = (out = NormalMeanVariance(1.0, 1.0), μ = NormalMeanVariance(2.0, 0.5)), q = (v = PointMass(2.0),)), false),
    ("NMV:(out,μ):inverse-gamma-v", NormalMeanVariance, (:out, :μ), (m = (out = NormalMeanVariance(1.0, 1.0), μ = NormalMeanVariance(2.0, 0.5)), q = (v = INVERSE_GAMMA,)), true),
    ("NMP:(out,μ):gamma-τ", NormalMeanPrecision, (:out, :μ), (m = (out = NormalMeanVariance(1.0, 1.0), μ = NormalWeightedMeanPrecision(2.0, 0.5)), q = (τ = GammaShapeRate(3.0, 2.0),)), false),
    ("NMV:(out,μ):point-mass-out", NormalMeanVariance, (:out, :μ), (m = (out = PointMass(1.0), μ = NormalMeanVariance(0.0, 1.0)), q = (v = PointMass(2.0),)), false),
    ("NMV:(out,μ):point-mass-μ", NormalMeanVariance, (:out, :μ), (m = (out = NormalMeanVariance(0.0, 1.0), μ = PointMass(1.0)), q = (v = PointMass(2.0),)), false),
    ("NMV:(out,μ):point-mass-out-inverse-gamma-v", NormalMeanVariance, (:out, :μ), (m = (out = PointMass(1.0), μ = NormalMeanVariance(0.0, 1.0)), q = (v = INVERSE_GAMMA,)), true),
    ("NMV:(out,μ):point-mass-μ-inverse-gamma-v", NormalMeanVariance, (:out, :μ), (m = (out = NormalMeanVariance(0.0, 1.0), μ = PointMass(1.0)), q = (v = INVERSE_GAMMA,)), true),
    ("NMV:(out,μ,v):point-mass-μ", NormalMeanVariance, (:out, :μ, :v), (m = (out = NormalMeanVariance(0.0, 1.0), μ = PointMass(1.0), v = PointMass(2.0)),), false),
    ("NMV:(out,μ,v):point-mass-out", NormalMeanVariance, (:out, :μ, :v), (m = (out = PointMass(1.0), μ = NormalMeanVariance(0.0, 1.0), v = PointMass(2.0)),), false),
    ("NMV:(out,μ,v):normals", NormalMeanVariance, (:out, :μ, :v), (m = (out = NormalMeanVariance(1.0, 1.0), μ = NormalMeanVariance(2.0, 0.5), v = PointMass(2.0)),), false),
    ("NMP:(out,μ):point-mass-out", NormalMeanPrecision, (:out, :μ), (m = (out = PointMass(1.0), μ = NormalMeanPrecision(0.0, 1.0)), q = (τ = GammaShapeRate(3.0, 2.0),)), false),
    ("NMP:(out,μ):point-mass-μ", NormalMeanPrecision, (:out, :μ), (m = (out = NormalMeanPrecision(0.0, 1.0), μ = PointMass(1.0)), q = (τ = GammaShapeRate(3.0, 2.0),)), false),
    ("NMP:(out,μ,τ):point-mass-μ", NormalMeanPrecision, (:out, :μ, :τ), (m = (out = NormalMeanPrecision(0.0, 1.0), μ = PointMass(1.0), τ = PointMass(2.0)),), false),
    ("NMP:(out,μ,τ):point-mass-out", NormalMeanPrecision, (:out, :μ, :τ), (m = (out = PointMass(1.0), μ = NormalMeanPrecision(0.0, 1.0), τ = PointMass(2.0)),), false),
    ("NMP:(out,μ,τ):normals", NormalMeanPrecision, (:out, :μ, :τ), (m = (out = NormalMeanVariance(1.0, 1.0), μ = NormalMeanVariance(2.0, 0.5), τ = PointMass(0.5)),), false),
    ("GSR:(out,α,β)", GammaShapeRate, (:out, :α, :β), (m = (out = GammaShapeScale(2.0, 2.0), α = PointMass(2.0), β = PointMass(3.0)),), false),
    ("Categorical:(out,p):point-mass-out", Categorical, (:out, :p), (m = (out = PointMass([0.0, 1.0]), p = Dirichlet([2.0, 1.0])),), false),
    ("Categorical:(out,p):point-mass-p", Categorical, (:out, :p), (m = (out = Categorical([0.2, 0.8]), p = PointMass([0.3, 0.7])),), false),
    ("Dirichlet:(out,a)", Dirichlet, (:out, :a), (m = (out = Dirichlet([2.0, 3.0]), a = PointMass([3.0, 1.0])),), false),
    ("Beta:(out,a,b)", Beta, (:out, :a, :b), (m = (out = Beta(1.0, 2.0), a = PointMass(1.0), b = PointMass(2.0)),), false),
    ("Bernoulli:(out,p):point-mass-out", Bernoulli, (:out, :p), (m = (out = PointMass(1.0), p = Beta(2.0, 1.0)),), false),
    ("Bernoulli:(out,p):point-mass-p", Bernoulli, (:out, :p), (m = (out = Bernoulli(0.8), p = PointMass(0.4)),), false),
    ("Gamma:(out,α,θ)", Gamma, (:out, :α, :θ), (m = (out = Gamma(2.0, 1.0), α = PointMass(2.0), θ = PointMass(1.0)),), false),
    ("GammaInverse:(out,α,θ)", GammaInverse, (:out, :α, :θ), (m = (out = GammaInverse(1.0, 2.0), α = PointMass(1.0), θ = PointMass(2.0)),), false),
    ("Poisson:(out,l)", Poisson, (:out, :l), (m = (out = PointMass(1.0), l = Gamma(2.0, 1.0)),), false),
    ("MvNMC:(out,μ):point-mass-Σ", MvNormalMeanCovariance, (:out, :μ), (m = (out = MvNormalWeightedMeanPrecision([1.0, 2.0], I2), μ = MvNormalMeanCovariance([3.0, 4.0], [2.0 0.5; 0.5 1.0])), q = (Σ = PointMass(2 * I2),)), false),
    ("MvNMC:(out,μ):inverse-wishart-Σ", MvNormalMeanCovariance, (:out, :μ), (m = (out = MvNormalWeightedMeanPrecision([1.0, 2.0], I2), μ = MvNormalMeanCovariance([3.0, 4.0], [2.0 0.5; 0.5 1.0])), q = (Σ = INVERSE_WISHART,)), MVNMC_669),
    ("MvNMC:(out,μ):point-mass-out", MvNormalMeanCovariance, (:out, :μ), (m = (out = PointMass([1.0, 1.0]), μ = MvNormalWeightedMeanPrecision([3.0, 4.0], I2)), q = (Σ = PointMass(2 * I2),)), false),
    ("MvNMC:(out,μ):point-mass-μ-inverse-wishart-Σ", MvNormalMeanCovariance, (:out, :μ), (m = (out = MvNormalWeightedMeanPrecision([3.0, 4.0], I2), μ = PointMass([1.0, 1.0])), q = (Σ = INVERSE_WISHART,)), MVNMC_669),
    ("MvNMC:(out,μ,Σ):point-mass-μ", MvNormalMeanCovariance, (:out, :μ, :Σ), (m = (out = MvNormalWeightedMeanPrecision([1.0, 2.0], I2), μ = PointMass([1.0, 1.0]), Σ = PointMass(2 * I2)),), false),
    ("MvNMC:(out,μ,Σ):point-mass-out", MvNormalMeanCovariance, (:out, :μ, :Σ), (m = (out = PointMass([1.0, 1.0]), μ = MvNormalWeightedMeanPrecision([3.0, 4.0], I2), Σ = PointMass(2 * I2)),), false),
    ("MvNMC:(out,μ,Σ):normals", MvNormalMeanCovariance, (:out, :μ, :Σ), (m = (out = MvNormalWeightedMeanPrecision([1.0, 2.0], I2), μ = MvNormalMeanCovariance([3.0, 4.0], [2.0 0.5; 0.5 1.0]), Σ = PointMass(2 * I2)),), false),
    ("MvNMP:(out,μ):wishart-Λ", MvNormalMeanPrecision, (:out, :μ), (m = (out = MvNormalWeightedMeanPrecision([1.0, 2.0], I2), μ = MvNormalMeanCovariance([3.0, 4.0], [2.0 0.5; 0.5 1.0])), q = (Λ = WISHART,)), false),
    ("MvNMP:(out,μ):point-mass-out", MvNormalMeanPrecision, (:out, :μ), (m = (out = PointMass([1.0, 1.0]), μ = MvNormalWeightedMeanPrecision([3.0, 4.0], I2)), q = (Λ = WISHART,)), false),
    ("MvNMP:(out,μ):point-mass-μ", MvNormalMeanPrecision, (:out, :μ), (m = (out = MvNormalWeightedMeanPrecision([3.0, 4.0], I2), μ = PointMass([1.0, 1.0])), q = (Λ = PointMass(I2 / 2),)), false),
    ("MvNMP:(out,μ,Λ):point-mass-μ", MvNormalMeanPrecision, (:out, :μ, :Λ), (m = (out = MvNormalWeightedMeanPrecision([1.0, 2.0], I2), μ = PointMass([1.0, 1.0]), Λ = PointMass(I2 / 2)),), false),
    ("MvNMP:(out,μ,Λ):point-mass-out", MvNormalMeanPrecision, (:out, :μ, :Λ), (m = (out = PointMass([1.0, 1.0]), μ = MvNormalWeightedMeanPrecision([3.0, 4.0], I2), Λ = PointMass(I2 / 2)),), false),
    ("MvNMP:(out,μ,Λ):normals", MvNormalMeanPrecision, (:out, :μ, :Λ), (m = (out = MvNormalWeightedMeanPrecision([1.0, 2.0], I2), μ = MvNormalMeanCovariance([3.0, 4.0], [2.0 0.5; 0.5 1.0]), Λ = PointMass(I2 / 2)),), false),
    ("MvNWMP:(out,ξ,Λ)", MvNormalWeightedMeanPrecision, (:out, :ξ, :Λ), (m = (out = MvNormalWeightedMeanPrecision([1.0, 2.0], I2), ξ = PointMass([0.5, 0.5]), Λ = PointMass(2 * I2)),), false),
    ("AND:(in1,in2)", AND, (:in1, :in2), (m = (out = Bernoulli(0.2), in1 = Bernoulli(0.8), in2 = Bernoulli(0.4)),), false),
    ("OR:(in1,in2)", OR, (:in1, :in2), (m = (out = Bernoulli(0.2), in1 = Bernoulli(0.8), in2 = Bernoulli(0.4)),), false),
    ("IMPLY:(in1,in2)", IMPLY, (:in1, :in2), (m = (out = Bernoulli(0.2), in1 = Bernoulli(0.8), in2 = Bernoulli(0.4)),), false),
]

const AVERAGE_ENERGY_CASES = [
    ("NMV:energy:singles", NormalMeanVariance, (q = (out = NormalMeanVariance(0.0, 1.0), μ = NormalMeanVariance(1.0, 2.0), v = INVERSE_GAMMA),), false),
    ("NMV:energy:joint", NormalMeanVariance, (q = (v = PointMass(2.0),), clusters = ((:out, :μ) => MvNormalMeanCovariance([0.0, 1.0], [1.0 0.2; 0.2 2.0]),)), false),
    ("NMP:energy:singles", NormalMeanPrecision, (q = (out = PointMass(1.2), μ = NormalWeightedMeanPrecision(1.0, 2.0), τ = GammaShapeRate(3.0, 2.0)),), false),
    ("GSR:energy:point-mass-α", GammaShapeRate, (q = (out = GammaShapeRate(2.0, 3.0), α = PointMass(2.0), β = GammaShapeRate(3.0, 1.5)),), false),
    ("GSR:energy:gamma-α", GammaShapeRate, (q = (out = GammaShapeRate(2.0, 3.0), α = GammaShapeRate(4.0, 2.0), β = PointMass(3.0)),), false),
    ("Categorical:energy", Categorical, (q = (out = Categorical([0.3, 0.2, 0.5]), p = Dirichlet([1.0, 3.0, 0.5])),), false),
    ("Dirichlet:energy", Dirichlet, (q = (out = Dirichlet([2.0, 3.0]), a = PointMass([2.0, 1.0])),), false),
    ("MvNMC:energy:point-mass-Σ", MvNormalMeanCovariance, (q = (out = PointMass([1.0, 1.0]), μ = MvNormalMeanPrecision([1.0, 1.0], I2), Σ = PointMass(2 * I2)),), false),
    ("MvNMC:energy:inverse-wishart-Σ", MvNormalMeanCovariance, (q = (out = MvNormalMeanCovariance([0.5, 1.0], [1.0 0.2; 0.2 0.5]), μ = MvNormalMeanCovariance([1.0, 1.0], I2), Σ = INVERSE_WISHART),), false),
    ("MvNMC:energy:joint", MvNormalMeanCovariance, (q = (Σ = INVERSE_WISHART,), clusters = ((:out, :μ) => MvNormalMeanCovariance([1.0, 0.5, 2.0, 1.0], [2.0 0.1 0.2 0.0; 0.1 1.5 0.0 0.3; 0.2 0.0 1.0 0.1; 0.0 0.3 0.1 2.5]),)), false),
    ("MvNMP:energy:wishart-Λ", MvNormalMeanPrecision, (q = (out = MvNormalMeanCovariance([0.5, 1.0], [1.0 0.2; 0.2 0.5]), μ = MvNormalMeanCovariance([1.0, 1.0], I2), Λ = WISHART),), false),
    ("MvNMP:energy:point-mass-Λ", MvNormalMeanPrecision, (q = (out = PointMass([1.0, 1.0]), μ = MvNormalMeanPrecision([1.0, 1.0], I2), Λ = PointMass(2 * I2)),), false),
    ("MvNMP:energy:joint-wishart", MvNormalMeanPrecision, (q = (Λ = WISHART,), clusters = ((:out, :μ) => MvNormalMeanCovariance([1.0, 0.5, 2.0, 1.0], [2.0 0.1 0.2 0.0; 0.1 1.5 0.0 0.3; 0.2 0.0 1.0 0.1; 0.0 0.3 0.1 2.5]),)), false),
    ("MvNMP:energy:joint-point-mass", MvNormalMeanPrecision, (q = (Λ = PointMass(I2 / 2),), clusters = ((:out, :μ) => MvNormalMeanCovariance([1.0, 0.5, 2.0, 1.0], [2.0 0.1 0.2 0.0; 0.1 1.5 0.0 0.3; 0.2 0.0 1.0 0.1; 0.0 0.3 0.1 2.5]),)), false),
    ("MvNWMP:energy", MvNormalWeightedMeanPrecision, (q = (out = MvNormalMeanCovariance([0.2, 0.4], [1.0 0.1; 0.1 0.7]), ξ = MvNormalMeanCovariance([0.5, -1.0], I2), Λ = PointMass([2.0 0.3; 0.3 1.5])),), false),
    ("Beta:energy", Beta, (q = (out = Beta(2.0, 3.0), a = PointMass(1.5), b = PointMass(2.5)),), false),
    ("Bernoulli:energy", Bernoulli, (q = (out = Bernoulli(0.3), p = Beta(2.0, 3.0)),), false),
    ("Gamma:energy:point-mass-α", Gamma, (q = (out = Gamma(2.0, 1.5), α = PointMass(2.0), θ = PointMass(1.0)),), false),
    ("Gamma:energy:gamma-α", Gamma, (q = (out = Gamma(2.0, 1.5), α = GammaShapeRate(4.0, 2.0), θ = PointMass(1.0)),), false),
    ("Gamma:energy:gamma-θ", Gamma, (q = (out = Gamma(3.0, 2.0), α = PointMass(2.0), θ = Gamma(3.0, 2.0)),), GAMMA_ENERGY),
    ("GammaInverse:energy", GammaInverse, (q = (out = GammaInverse(2.0, 1.0), α = PointMass(2.0), θ = PointMass(1.0)),), GAMMA_INVERSE_ENERGY),
    ("HalfNormal:energy", HalfNormal, (q = (out = GammaShapeRate(2.0, 1.0), v = PointMass(2.0)),), false),
    ("Poisson:energy:point-masses", Poisson, (q = (out = PointMass(3), l = PointMass(2.0)),), false),
    ("Poisson:energy:poisson-out", Poisson, (q = (out = Poisson(4.0), l = GammaShapeRate(2.0, 1.0)),), false),
    ("Uniform:energy", Uniform, (q = (out = Beta(0.3, 0.7), a = PointMass(0.0), b = PointMass(1.0)),), false),
    ("Uninformative:energy", Uninformative, (q = (out = NormalMeanVariance(0.0, 1.0),),), false),
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

# `flagged` is `false`, `true` for #669, or the reason of another declared correction.
declared_reason(flagged::Bool) = NMV_669
declared_reason(flagged::AbstractString) = flagged
declare(id, flagged) = flagged === false ? DeclaredDisagreement[] : [DeclaredDisagreement(id; kind = :correction, reasoning = declared_reason(flagged))]

@testset "StandardMessagePassingRules against v6" begin
    @testset "message rules" begin
        for (id, node, edge, inputs, flagged) in MESSAGE_CASES
            m, q = get(inputs, :m, NamedTuple()), get(inputs, :q, NamedTuple())
            store = AnnotationStore()
            v7 = call_message_update_rule(node, edge; m, q, ann = store)
            v6, v6_logscale = v6_message_update(v6_node(node), edge, m, q)
            record = compare_with_reference(id, v7, as_v7(node, v6); inputs, node = string(node), target = ":$edge", v7_logscale = getannotation(store, :logscale, nothing), v6_logscale, declared = declare(id, flagged))
            # A declared correction must actually differ, or the declaration is stale.
            @test (flagged !== false) == (record.outcome === :correction)
        end
    end
    @testset "message rules consuming a joint" begin
        for (id, node, edge, clusters, flagged) in CLUSTER_MESSAGE_CASES
            v7 = call_message_update_rule(node, edge; clusters)
            v6, _ = v6_message_update(node, edge, NamedTuple(), NamedTuple{map(V6Oracle.v6_name, Tuple(first.(clusters)))}(Tuple(last.(clusters))))
            record = compare_with_reference(id, v7, v6; inputs = clusters, node = string(node), target = ":$edge", declared = declare(id, flagged))
            @test (flagged !== false) == (record.outcome === :correction)
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
    # Belief propagation towards NMV's `v` is a log-density on the half line with no family;
    # the port and v6 are compared by evaluating it.
    @testset "log-density messages" begin
        for m in [
                (out = PointMass(2.0), μ = NormalMeanVariance(0.0, 1.0)),
                (out = NormalMeanVariance(0.5, 2.0), μ = PointMass(-3.5)),
                (out = NormalMeanVariance(1.0, 0.5), μ = NormalMeanVariance(-1.0, 2.0)),
            ]
            v7 = call_message_update_rule(NormalMeanVariance, :v; m)
            v6, _ = v6_message_update(NormalMeanVariance, :v, m, NamedTuple())
            @test all(v -> logpdf(v7, v) ≈ logpdf(v6, v), (0.1, 1.0, 3.5, 10.0))
        end
    end
    @testset "marginal rules" begin
        for (id, node, members, inputs, flagged) in MARGINAL_CASES
            m, q = get(inputs, :m, NamedTuple()), get(inputs, :q, NamedTuple())
            v7 = call_marginal_update_rule(node, members; m, q)
            v6 = v6_marginal_update(v6_node(node), members, m, q)
            record = compare_with_reference(id, v7, as_v7(node, v6); inputs, node = string(node), target = string(members), declared = declare(id, flagged))
            @test (flagged !== false) == (record.outcome === :correction)
        end
    end
    @testset "average energies" begin
        for (id, node, inputs, flagged) in AVERAGE_ENERGY_CASES
            q, clusters = get(inputs, :q, NamedTuple()), get(inputs, :clusters, ())
            v7 = call_average_energy(node; q, clusters)
            v6 = v6_average_energy(v6_node(node), q, clusters)
            record = compare_with_reference(id, v7, v6; inputs, node = string(node), target = "energy", declared = declare(id, flagged))
            @test (flagged !== false) == (record.outcome === :correction)
        end
    end
end
