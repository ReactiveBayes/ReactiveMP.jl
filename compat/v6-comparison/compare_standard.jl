# The rules ported into StandardMessagePassingRules, compared with their v6 originals on
# identical inputs. A difference fails unless it is declared, with its reasoning.
#
#   julia --startup-file=no --project=compat/v6-comparison compat/v6-comparison/compare_standard.jl

include(joinpath(@__DIR__, "V6Oracle.jl"))

using .V6Oracle, Test
using ExponentialFamily, BayesBase, Distributions
using MessagePassingRulesBase, MessagePassingRulesTestUtils, StandardMessagePassingRules
using MessagePassingRulesBase: getlogscale, isdefined_logscale
using LinearAlgebra: dot
import LinearAlgebra, Random
using MessagePassingRulesBase: RuleContext

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
# v6's variational Wishart `:out` rules take the inverse scale a `q_S` contributes as E[S]⁻¹;
# naive VMP gives E[S⁻¹], which v6's own Wishart energy uses. They agree for a point mass.
const WISHART_OUT = "ReactiveMP.jl#675: v6 uses E[S]⁻¹ for the inverse scale a non-point-mass q_S contributes to Wishart's out; naive VMP gives E[S⁻¹], as v6's own Wishart energy does"
# v6's variational MvNormalGamma `:out` rule passes E[β] as the rate; the expectation of
# γ (out - μ)ᵀΛ(out - μ)/2 under a `q_μ` adds tr(E[Λ] Cov μ)/2 to it. They agree for a point mass.
const MVNG_OUT = "ReactiveMP.jl#676: v6's MvNormalGamma out rule drops tr(E[Λ] Cov μ)/2 from the rate for a non-point-mass q_μ"
# v6's `+` `:in1` specialisation for two BLAS-typed MvNormalWeightedMeanPrecision messages
# takes μ_in2 - μ_out, the generic rule's negation of the mean; `-` reaches it through its redirects.
const ADDITION_SIGN = "ReactiveMP.jl#677: v6's `+` in1 rule for two weighted-mean normals computes E[in2] - E[out]; it is E[out] - E[in2], as v6's generic rule gives"
# v6's `-` marginal shifts in2 by out - in1 when in1 is known; out = in1 - in2 gives in1 - out.
const SUBTRACTION_MARGINAL = "ReactiveMP.jl#678: v6's `-` marginal takes in2's likelihood at out - in1 for a known in1; it is in1 - out, as v6's own `-` in2 message rule gives"
# v6's `*` `:in` log-scale for a scalar A is -logdet(a) = -log a; m(x) = N_out(a x) integrates
# to |a|^(-d), so it is -d log |a|. They agree for d = 1 and a > 0; v6 throws for a < 0.
const MULTIPLICATION_LOGSCALE = "ReactiveMP.jl#680: v6's `*` in log-scale for a scalar a is -log a; the message integrates to |a|^(-d), so it is -d log |a|"
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
    ("MvNMSP:out:q", MvNormalMeanScalePrecision, :out, (q = (μ = MvNormalMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), γ = Gamma(3.0, 2.0)),), false),
    ("MvNMSP:out:m-normal", MvNormalMeanScalePrecision, :out, (m = (μ = MvNormalMeanCovariance([0.0, 1.0], [2.0 -1.0; -1.0 4.0]),), q = (γ = GammaShapeRate(3.0, 2.0),)), false),
    ("MvNMSP:out:m-scale-precision", MvNormalMeanScalePrecision, :out, (m = (μ = MvNormalMeanScalePrecision([2.0, 1.0], 3.0),), q = (γ = GammaShapeRate(2.0, 1.0),)), false),
    ("MvNMSP:μ:m-normal", MvNormalMeanScalePrecision, :μ, (m = (out = MvNormalWeightedMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]),), q = (γ = GammaShapeRate(2.0, 1.0),)), false),
    ("MvNMSP:γ:q", MvNormalMeanScalePrecision, :γ, (q = (out = MvNormalMeanCovariance([1.0, 2.0], [3.0 2.0; 2.0 4.0]), μ = MvNormalMeanPrecision([3.0, 5.0], [3.0 2.0; 2.0 4.0])),), false),
    ("MvNMSMP:out:q", MvNormalMeanScaleMatrixPrecision, :out, (q = (μ = MvNormalMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), γ = Gamma(3.0, 2.0), G = Wishart(4.0, [3.0 0.5; 0.5 2.0])),), false),
    ("MvNMSMP:out:m-normal", MvNormalMeanScaleMatrixPrecision, :out, (m = (μ = MvNormalMeanCovariance([0.0, 1.0], [2.0 -1.0; -1.0 4.0]),), q = (γ = GammaShapeRate(3.0, 2.0), G = Wishart(4.0, [3.0 0.5; 0.5 2.0]))), false),
    ("MvNMSMP:μ:q", MvNormalMeanScaleMatrixPrecision, :μ, (q = (out = MvNormalWeightedMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), γ = GammaShapeRate(2.0, 1.0), G = Wishart(4.0, [3.0 0.5; 0.5 2.0])),), false),
    ("MvNMSMP:μ:m-normal", MvNormalMeanScaleMatrixPrecision, :μ, (m = (out = MvNormalWeightedMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]),), q = (γ = GammaShapeRate(2.0, 1.0), G = Wishart(4.0, [3.0 0.5; 0.5 2.0]))), false),
    ("MvNMSMP:γ:q", MvNormalMeanScaleMatrixPrecision, :γ, (q = (out = MvNormalMeanCovariance([1.0, 2.0], [3.0 2.0; 2.0 4.0]), μ = MvNormalMeanPrecision([3.0, 5.0], [3.0 2.0; 2.0 4.0]), G = Wishart(4.0, [3.0 0.5; 0.5 2.0])),), false),
    ("MvNMSMP:G:q", MvNormalMeanScaleMatrixPrecision, :G, (q = (out = MvNormalMeanCovariance([1.0, 2.0], [3.0 2.0; 2.0 4.0]), μ = MvNormalMeanPrecision([3.0, 5.0], [3.0 2.0; 2.0 4.0]), γ = GammaShapeRate(2.0, 4.0)),), false),
    ("Wishart:out:m-point-masses", Wishart, :out, (m = (ν = PointMass(3.0), S = PointMass([2.0 0.3; 0.3 1.5])),), false),
    ("Wishart:out:q-ν", Wishart, :out, (q = (ν = PointMass(3.0),), m = (S = PointMass([2.0 0.3; 0.3 1.5]),)), false),
    ("Wishart:out:q-point-masses", Wishart, :out, (q = (ν = PointMass(3.0), S = PointMass([2.0 0.3; 0.3 1.5])),), false),
    ("Wishart:out:q-inverse-wishart-S", Wishart, :out, (m = (ν = PointMass(3.0),), q = (S = INVERSE_WISHART,)), WISHART_OUT),
    ("Wishart:out:q-ν-q-inverse-wishart-S", Wishart, :out, (q = (ν = PointMass(3.0), S = INVERSE_WISHART),), WISHART_OUT),
    ("InverseWishart:out:m-point-masses", InverseWishart, :out, (m = (ν = PointMass(5.0), S = PointMass([2.0 0.3; 0.3 1.5])),), false),
    ("InverseWishart:out:q-ν", InverseWishart, :out, (q = (ν = PointMass(5.0),), m = (S = PointMass([2.0 0.3; 0.3 1.5]),)), false),
    ("InverseWishart:out:q-wishart-S", InverseWishart, :out, (m = (ν = PointMass(5.0),), q = (S = WISHART,)), false),
    ("InverseWishart:out:q-ν-q-wishart-S", InverseWishart, :out, (q = (ν = PointMass(5.0), S = WISHART),), false),
    ("DirichletCollection:out:m", DirichletCollection, :out, (m = (a = PointMass([1.0 2.0; 3.0 0.5; 2.0 1.5]),),), false),
    ("DirichletCollection:out:q", DirichletCollection, :out, (q = (a = PointMass([1.0 2.0; 3.0 0.5; 2.0 1.5]),),), false),
    ("MvNormalGamma:out:m-point-masses", MvNormalGamma, :out, (m = (μ = PointMass([0.5, -1.0]), Λ = PointMass([2.0 0.3; 0.3 1.5]), α = PointMass(2.0), β = PointMass(3.0)),), false),
    ("MvNormalGamma:out:q-point-masses", MvNormalGamma, :out, (q = (μ = PointMass([0.5, -1.0]), Λ = PointMass([2.0 0.3; 0.3 1.5]), α = PointMass(2.0), β = PointMass(3.0)),), false),
    ("MvNormalGamma:out:q-gamma-α-β", MvNormalGamma, :out, (q = (μ = PointMass([0.5, -1.0]), Λ = WISHART, α = GammaShapeRate(4.0, 2.0), β = GammaShapeRate(6.0, 2.0)),), false),
    ("MvNormalGamma:out:q-normal-μ", MvNormalGamma, :out, (q = (μ = MvNormalMeanCovariance([0.5, -1.0], [1.0 0.2; 0.2 2.0]), Λ = PointMass([2.0 0.3; 0.3 1.5]), α = PointMass(2.0), β = PointMass(3.0)),), MVNG_OUT),
    ("MvNormalWishart:out:q-point-masses", MvNormalWishart, :out, (q = (μ = PointMass([0.5, -1.0]), W = PointMass([2.0 0.3; 0.3 1.5]), λ = PointMass(2.5), ν = PointMass(4.0)),), false),
    ("MatrixNormal:out:m-point-masses", MatrixNormal, :out, (m = (M = PointMass([0.5 1.0; 2.0 3.0; 4.0 5.0]), U = PointMass([2.0 0.3 0.0; 0.3 3.0 0.1; 0.0 0.1 4.0]), V = PointMass([1.0 0.5; 0.5 2.0])),), false),
    ("MatrixNormal:out:m-matrix-normal", MatrixNormal, :out, (m = (M = MatrixNormal([0.5 1.0; 2.0 3.0; 4.0 5.0], [0.5 0.1 0.0; 0.1 0.5 0.0; 0.0 0.0 0.5], [1.0 0.2; 0.2 0.5]), U = PointMass([2.0 0.3 0.0; 0.3 3.0 0.1; 0.0 0.1 4.0]), V = PointMass([1.0 0.5; 0.5 2.0])),), false),
    ("MatrixNormal:out:m-inverse-wishart-U", MatrixNormal, :out, (m = (M = PointMass([0.5 1.0; 2.0 3.0; 4.0 5.0]), U = InverseWishart(7.0, [2.0 0.3 0.0; 0.3 3.0 0.1; 0.0 0.1 4.0]), V = PointMass([1.0 0.5; 0.5 2.0])),), false),
    ("MatrixNormal:out:m-inverse-wishart-V", MatrixNormal, :out, (m = (M = PointMass([0.5 1.0; 2.0 3.0; 4.0 5.0]), U = PointMass([2.0 0.3 0.0; 0.3 3.0 0.1; 0.0 0.1 4.0]), V = InverseWishart(6.0, [1.0 0.5; 0.5 2.0])),), false),
    ("MatrixNormal:out:q", MatrixNormal, :out, (q = (M = MatrixNormal([0.5 1.0; 2.0 3.0; 4.0 5.0], [0.5 0.1 0.0; 0.1 0.5 0.0; 0.0 0.0 0.5], [1.0 0.2; 0.2 0.5]), U = InverseWishart(7.0, [2.0 0.3 0.0; 0.3 3.0 0.1; 0.0 0.1 4.0]), V = InverseWishart(6.0, [1.0 0.5; 0.5 2.0])),), false),
    ("MatrixNormal:M:m-point-masses", MatrixNormal, :M, (m = (out = PointMass([0.5 1.0; 2.0 3.0; 4.0 5.0]), U = PointMass([2.0 0.3 0.0; 0.3 3.0 0.1; 0.0 0.1 4.0]), V = PointMass([1.0 0.5; 0.5 2.0])),), false),
    ("MatrixNormal:M:m-matrix-normal", MatrixNormal, :M, (m = (out = MatrixNormal([0.5 1.0; 2.0 3.0; 4.0 5.0], [0.5 0.1 0.0; 0.1 0.5 0.0; 0.0 0.0 0.5], [1.0 0.2; 0.2 0.5]), U = PointMass([2.0 0.3 0.0; 0.3 3.0 0.1; 0.0 0.1 4.0]), V = PointMass([1.0 0.5; 0.5 2.0])),), false),
    ("MatrixNormal:M:m-inverse-wishart-U", MatrixNormal, :M, (m = (out = PointMass([0.5 1.0; 2.0 3.0; 4.0 5.0]), U = InverseWishart(7.0, [2.0 0.3 0.0; 0.3 3.0 0.1; 0.0 0.1 4.0]), V = PointMass([1.0 0.5; 0.5 2.0])),), false),
    ("MatrixNormal:M:m-inverse-wishart-V", MatrixNormal, :M, (m = (out = PointMass([0.5 1.0; 2.0 3.0; 4.0 5.0]), U = PointMass([2.0 0.3 0.0; 0.3 3.0 0.1; 0.0 0.1 4.0]), V = InverseWishart(6.0, [1.0 0.5; 0.5 2.0])),), false),
    ("MatrixNormal:M:q", MatrixNormal, :M, (q = (out = MatrixNormal([0.5 1.0; 2.0 3.0; 4.0 5.0], [0.5 0.1 0.0; 0.1 0.5 0.0; 0.0 0.0 0.5], [1.0 0.2; 0.2 0.5]), U = InverseWishart(7.0, [2.0 0.3 0.0; 0.3 3.0 0.1; 0.0 0.1 4.0]), V = InverseWishart(6.0, [1.0 0.5; 0.5 2.0])),), false),
    ("MatrixNormal:U:m-point-masses", MatrixNormal, :U, (m = (out = PointMass([1.0 2.0; 3.0 4.0; 5.0 6.0]), M = PointMass([0.5 1.0; 2.0 3.0; 4.0 5.0]), V = PointMass([1.0 0.5; 0.5 2.0])),), false),
    ("MatrixNormal:U:q", MatrixNormal, :U, (q = (out = MatrixNormal([0.5 1.0; 2.0 3.0; 4.0 5.0], [0.5 0.1 0.0; 0.1 0.5 0.0; 0.0 0.0 0.5], [1.0 0.2; 0.2 0.5]), M = MatrixNormal([0.5 1.0; 2.0 3.0; 4.0 5.0], [0.5 0.1 0.0; 0.1 0.5 0.0; 0.0 0.0 0.5], [1.0 0.2; 0.2 0.5]), V = InverseWishart(6.0, [1.0 0.5; 0.5 2.0])),), false),
    ("MatrixNormal:V:m-point-masses", MatrixNormal, :V, (m = (out = PointMass([1.0 2.0; 3.0 4.0; 5.0 6.0]), M = PointMass([0.5 1.0; 2.0 3.0; 4.0 5.0]), U = PointMass([2.0 0.3 0.0; 0.3 3.0 0.1; 0.0 0.1 4.0])),), false),
    ("MatrixNormal:V:q", MatrixNormal, :V, (q = (out = MatrixNormal([0.5 1.0; 2.0 3.0; 4.0 5.0], [0.5 0.1 0.0; 0.1 0.5 0.0; 0.0 0.0 0.5], [1.0 0.2; 0.2 0.5]), M = PointMass([0.5 1.0; 2.0 3.0; 4.0 5.0]), U = InverseWishart(7.0, [2.0 0.3 0.0; 0.3 3.0 0.1; 0.0 0.1 4.0])),), false),
    ("MatrixNormalWishart:out:q-point-masses", MatrixNormalWishart, :out, (q = (M = PointMass([0.5 1.0; 2.0 3.0; 4.0 5.0]), U = PointMass([2.0 0.3 0.0; 0.3 3.0 0.1; 0.0 0.1 4.0]), V = PointMass([1.0 0.5; 0.5 2.0]), ν = PointMass(4.0)),), false),
    ("+:out:normals", +, :out, (m = (in1 = NormalMeanVariance(1.0, 2.0), in2 = NormalMeanPrecision(3.0, 0.5)),), false),
    ("+:out:normal-point-mass", +, :out, (m = (in1 = NormalMeanPrecision(3.0, 0.5), in2 = PointMass(2.0)),), false),
    ("+:out:point-mass-normal", +, :out, (m = (in1 = PointMass(2.0), in2 = NormalMeanVariance(1.0, 2.0)),), false),
    ("+:out:mv-normals", +, :out, (m = (in1 = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.3; 0.3 1.0]), in2 = MvNormalMeanCovariance([1.0, -1.0], [1.0 0.2; 0.2 2.0])),), false),
    ("+:out:mv-weighted-point-mass", +, :out, (m = (in1 = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.3; 0.3 1.0]), in2 = PointMass([1.0, 2.0])),), false),
    ("+:out:point-mass-mv-weighted", +, :out, (m = (in1 = PointMass([1.0, 2.0]), in2 = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.3; 0.3 1.0])),), false),
    ("+:in1:normals", +, :in1, (m = (out = NormalMeanVariance(1.0, 2.0), in2 = NormalMeanPrecision(3.0, 0.5)),), false),
    ("+:in1:normal-point-mass", +, :in1, (m = (out = NormalMeanPrecision(3.0, 0.5), in2 = PointMass(2.0)),), false),
    ("+:in1:point-mass-normal", +, :in1, (m = (out = PointMass(2.0), in2 = NormalMeanVariance(1.0, 2.0)),), false),
    ("+:in1:mv-normals", +, :in1, (m = (out = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.3; 0.3 1.0]), in2 = MvNormalMeanCovariance([1.0, -1.0], [1.0 0.2; 0.2 2.0])),), false),
    ("+:in1:mv-weighted-point-mass", +, :in1, (m = (out = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.3; 0.3 1.0]), in2 = PointMass([1.0, 2.0])),), false),
    ("+:in1:point-mass-mv-weighted", +, :in1, (m = (out = PointMass([1.0, 2.0]), in2 = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.3; 0.3 1.0])),), false),
    ("+:in2:normals", +, :in2, (m = (out = NormalMeanVariance(1.0, 2.0), in1 = NormalMeanPrecision(3.0, 0.5)),), false),
    ("+:in2:normal-point-mass", +, :in2, (m = (out = NormalMeanPrecision(3.0, 0.5), in1 = PointMass(2.0)),), false),
    ("+:in2:point-mass-normal", +, :in2, (m = (out = PointMass(2.0), in1 = NormalMeanVariance(1.0, 2.0)),), false),
    ("+:in2:mv-normals", +, :in2, (m = (out = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.3; 0.3 1.0]), in1 = MvNormalMeanCovariance([1.0, -1.0], [1.0 0.2; 0.2 2.0])),), false),
    ("+:in2:mv-weighted-point-mass", +, :in2, (m = (out = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.3; 0.3 1.0]), in1 = PointMass([1.0, 2.0])),), false),
    ("+:in2:point-mass-mv-weighted", +, :in2, (m = (out = PointMass([1.0, 2.0]), in1 = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.3; 0.3 1.0])),), false),
    ("+:in1:two-weighted", +, :in1, (m = (out = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.3; 0.3 1.0]), in2 = MvNormalWeightedMeanPrecision([1.0, 1.0], [1.0 0.0; 0.0 1.5])),), ADDITION_SIGN),
    ("+:in2:two-weighted", +, :in2, (m = (out = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.3; 0.3 1.0]), in1 = MvNormalWeightedMeanPrecision([1.0, 1.0], [1.0 0.0; 0.0 1.5])),), ADDITION_SIGN),
    ("+:out:two-weighted", +, :out, (m = (in1 = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.3; 0.3 1.0]), in2 = MvNormalWeightedMeanPrecision([1.0, 1.0], [1.0 0.0; 0.0 1.5])),), false),
    ("+:out:point-masses", +, :out, (m = (in1 = PointMass(1.0), in2 = PointMass(2.5)),), false),
    ("-:out:normals", -, :out, (m = (in1 = NormalMeanVariance(1.0, 2.0), in2 = NormalMeanPrecision(3.0, 0.5)),), false),
    ("-:out:normal-point-mass", -, :out, (m = (in1 = NormalMeanPrecision(3.0, 0.5), in2 = PointMass(2.0)),), false),
    ("-:out:point-mass-normal", -, :out, (m = (in1 = PointMass(2.0), in2 = NormalMeanVariance(1.0, 2.0)),), false),
    ("-:out:mv-normals", -, :out, (m = (in1 = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.3; 0.3 1.0]), in2 = MvNormalMeanCovariance([1.0, -1.0], [1.0 0.2; 0.2 2.0])),), false),
    ("-:out:mv-weighted-point-mass", -, :out, (m = (in1 = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.3; 0.3 1.0]), in2 = PointMass([1.0, 2.0])),), false),
    ("-:out:point-mass-mv-weighted", -, :out, (m = (in1 = PointMass([1.0, 2.0]), in2 = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.3; 0.3 1.0])),), false),
    ("-:in1:normals", -, :in1, (m = (out = NormalMeanVariance(1.0, 2.0), in2 = NormalMeanPrecision(3.0, 0.5)),), false),
    ("-:in1:normal-point-mass", -, :in1, (m = (out = NormalMeanPrecision(3.0, 0.5), in2 = PointMass(2.0)),), false),
    ("-:in1:point-mass-normal", -, :in1, (m = (out = PointMass(2.0), in2 = NormalMeanVariance(1.0, 2.0)),), false),
    ("-:in1:mv-normals", -, :in1, (m = (out = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.3; 0.3 1.0]), in2 = MvNormalMeanCovariance([1.0, -1.0], [1.0 0.2; 0.2 2.0])),), false),
    ("-:in1:mv-weighted-point-mass", -, :in1, (m = (out = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.3; 0.3 1.0]), in2 = PointMass([1.0, 2.0])),), false),
    ("-:in1:point-mass-mv-weighted", -, :in1, (m = (out = PointMass([1.0, 2.0]), in2 = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.3; 0.3 1.0])),), false),
    ("-:in2:normals", -, :in2, (m = (out = NormalMeanVariance(1.0, 2.0), in1 = NormalMeanPrecision(3.0, 0.5)),), false),
    ("-:in2:normal-point-mass", -, :in2, (m = (out = NormalMeanPrecision(3.0, 0.5), in1 = PointMass(2.0)),), false),
    ("-:in2:point-mass-normal", -, :in2, (m = (out = PointMass(2.0), in1 = NormalMeanVariance(1.0, 2.0)),), false),
    ("-:in2:mv-normals", -, :in2, (m = (out = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.3; 0.3 1.0]), in1 = MvNormalMeanCovariance([1.0, -1.0], [1.0 0.2; 0.2 2.0])),), false),
    ("-:in2:mv-weighted-point-mass", -, :in2, (m = (out = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.3; 0.3 1.0]), in1 = PointMass([1.0, 2.0])),), false),
    ("-:in2:point-mass-mv-weighted", -, :in2, (m = (out = PointMass([1.0, 2.0]), in1 = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.3; 0.3 1.0])),), false),
    ("-:out:two-weighted", -, :out, (m = (in1 = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.3; 0.3 1.0]), in2 = MvNormalWeightedMeanPrecision([1.0, 1.0], [1.0 0.0; 0.0 1.5])),), ADDITION_SIGN),
    ("-:in1:two-weighted", -, :in1, (m = (out = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.3; 0.3 1.0]), in2 = MvNormalWeightedMeanPrecision([1.0, 1.0], [1.0 0.0; 0.0 1.5])),), false),
    ("-:in2:two-weighted", -, :in2, (m = (out = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.3; 0.3 1.0]), in1 = MvNormalWeightedMeanPrecision([1.0, 1.0], [1.0 0.0; 0.0 1.5])),), ADDITION_SIGN),
    ("dot:out:point-mass-normal", dot, :out, (m = (in1 = PointMass(2.0), in2 = NormalMeanVariance(1.0, 2.0)),), false),
    ("dot:out:point-mass-mv-normal", dot, :out, (m = (in1 = PointMass([2.0, 0.5]), in2 = MvNormalMeanCovariance([1.0, -1.0], [1.0 0.2; 0.2 2.0])),), false),
    ("dot:out:mv-normal-point-mass", dot, :out, (m = (in1 = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.3; 0.3 1.0]), in2 = PointMass([2.0, 0.5])),), false),
    ("dot:in2:scalar", dot, :in2, (m = (out = NormalMeanVariance(1.0, 2.0), in1 = PointMass(2.0)),), false),
    ("dot:in2:vector", dot, :in2, (m = (out = NormalMeanPrecision(2.0, 3.0), in1 = PointMass([2.0, 0.5])),), false),
    ("dot:in1:vector", dot, :in1, (m = (out = NormalWeightedMeanPrecision(2.0, 3.0), in2 = PointMass([1.0, -0.5])),), false),
    ("*:out:scalar-normal", *, :out, (m = (A = PointMass(2.0), in = NormalMeanVariance(1.0, 2.0)),), false),
    ("*:out:scalar-mv-covariance", *, :out, (m = (A = PointMass(-1.5), in = MvNormalMeanCovariance([1.0, -1.0], [1.0 0.2; 0.2 2.0])),), false),
    ("*:out:scalar-mv-precision", *, :out, (m = (A = PointMass(0.5), in = MvNormalMeanPrecision([1.0, 2.0], [3.0 0.5; 0.5 2.0])),), false),
    ("*:out:scalar-mv-weighted", *, :out, (m = (A = PointMass(2.0), in = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.3; 0.3 1.0])),), false),
    ("*:out:normal-scalar", *, :out, (m = (A = MvNormalMeanCovariance([1.0, -1.0], [1.0 0.2; 0.2 2.0]), in = PointMass(2.0)),), false),
    ("*:out:uniform-scaling", *, :out, (m = (A = PointMass(2.0 * LinearAlgebra.I), in = MvNormalMeanPrecision([1.0, 2.0], [3.0 0.5; 0.5 2.0])),), false),
    ("*:out:matrix-normal", *, :out, (m = (A = PointMass([1.0 2.0; 0.5 2.0]), in = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.3; 0.3 1.0])),), false),
    ("*:out:vector-normal", *, :out, (m = (A = PointMass([2.0, 0.5]), in = NormalMeanVariance(1.0, 2.0)),), false),
    ("*:out:normal-vector", *, :out, (m = (A = NormalMeanVariance(1.0, 2.0), in = PointMass([2.0, 0.5])),), false),
    ("*:out:scalar-gamma", *, :out, (m = (A = PointMass(2.0), in = GammaShapeRate(3.0, 2.0)),), false),
    ("*:out:gamma-scalar", *, :out, (m = (A = GammaShapeRate(3.0, 2.0), in = PointMass(2.0)),), false),
    ("*:out:point-masses", *, :out, (m = (A = PointMass([1.0 2.0; 0.5 2.0]), in = PointMass([1.0, 2.0])),), false),
    ("*:in:normal-scalar", *, :in, (m = (out = NormalMeanVariance(1.0, 2.0), A = PointMass(2.0)),), false),
    ("*:in:mv-normal-scalar", *, :in, (m = (out = MvNormalMeanCovariance([1.0, -1.0], [1.0 0.2; 0.2 2.0]), A = PointMass(2.0)),), MULTIPLICATION_LOGSCALE),
    # No `*:in` case for a UniformScaling A: v6's rule for it takes its arguments in reverse order, which v6's dispatch never reaches.
    ("*:in:mv-weighted-matrix", *, :in, (m = (out = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.3; 0.3 1.0]), A = PointMass([1.0 2.0; 0.5 2.0])),), false),
    ("*:in:mv-covariance-matrix", *, :in, (m = (out = MvNormalMeanCovariance([1.0, -1.0], [1.0 0.2; 0.2 2.0]), A = PointMass([1.0 2.0; 0.5 2.0])),), false),
    ("*:in:mv-precision-vector", *, :in, (m = (out = MvNormalMeanPrecision([1.0, 2.0], [3.0 0.5; 0.5 2.0]), A = PointMass([2.0, 0.5])),), false),
    ("*:in:mv-covariance-vector", *, :in, (m = (out = MvNormalMeanCovariance([1.0, -1.0], [1.0 0.2; 0.2 2.0]), A = PointMass([2.0, 0.5])),), false),
    ("*:in:gamma-scalar", *, :in, (m = (out = GammaShapeRate(3.0, 2.0), A = PointMass(2.0)),), false),
    ("*:in:point-masses", *, :in, (m = (out = PointMass([3.0, 1.0]), A = PointMass([1.0 2.0; 0.5 2.0])),), false),
    ("*:A:normal-scalar", *, :A, (m = (out = NormalMeanVariance(1.0, 2.0), in = PointMass(2.0)),), false),
    ("*:A:mv-precision-vector", *, :A, (m = (out = MvNormalMeanPrecision([1.0, 2.0], [3.0 0.5; 0.5 2.0]), in = PointMass([2.0, 0.5])),), false),
    ("*:A:mv-covariance-vector", *, :A, (m = (out = MvNormalMeanCovariance([1.0, -1.0], [1.0 0.2; 0.2 2.0]), in = PointMass([2.0, 0.5])),), false),
    ("*:A:gamma-scalar", *, :A, (m = (out = GammaShapeRate(3.0, 2.0), in = PointMass(2.0)),), false),
    ("*:A:point-masses", *, :A, (m = (out = PointMass(6.0), in = PointMass(2.0)),), false),
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

# (id, node, target, clusters, flagged), then the marginals outside the joint, if any.
const CLUSTER_MESSAGE_CASES = [
    ("NMP:τ:q-joint", NormalMeanPrecision, :τ, ((:out, :μ) => MvNormalWeightedMeanPrecision([1.0, 0.5], [3.0 -1.0; -1.0 2.0]),), false),
    ("MvNMC:Σ:q-joint", MvNormalMeanCovariance, :Σ, ((:out, :μ) => MvNormalMeanCovariance([1.0, 0.5, 2.0, 1.0], [2.0 0.1 0.2 0.0; 0.1 1.5 0.0 0.3; 0.2 0.0 1.0 0.1; 0.0 0.3 0.1 2.5]),), false),
    ("MvNMP:Λ:q-joint", MvNormalMeanPrecision, :Λ, ((:out, :μ) => MvNormalMeanCovariance([1.0, 0.5, 2.0, 1.0], [2.0 0.1 0.2 0.0; 0.1 1.5 0.0 0.3; 0.2 0.0 1.0 0.1; 0.0 0.3 0.1 2.5]),), false),
    ("MvNMSP:γ:q-joint", MvNormalMeanScalePrecision, :γ, ((:out, :μ) => MvNormalMeanCovariance([1.0, 0.5, 2.0, 1.0], [2.0 0.1 0.2 0.0; 0.1 1.5 0.0 0.3; 0.2 0.0 1.0 0.1; 0.0 0.3 0.1 2.5]),), false),
    ("MvNMSMP:γ:q-joint", MvNormalMeanScaleMatrixPrecision, :γ, ((:out, :μ) => MvNormalMeanCovariance([1.0, 0.5, 2.0, 1.0], [2.0 0.1 0.2 0.0; 0.1 1.5 0.0 0.3; 0.2 0.0 1.0 0.1; 0.0 0.3 0.1 2.5]),), false, (G = Wishart(4.0, [3.0 0.5; 0.5 2.0]),)),
    ("MvNMSMP:G:q-joint", MvNormalMeanScaleMatrixPrecision, :G, ((:out, :μ) => MvNormalMeanCovariance([1.0, 0.5, 2.0, 1.0], [2.0 0.1 0.2 0.0; 0.1 1.5 0.0 0.3; 0.2 0.0 1.0 0.1; 0.0 0.3 0.1 2.5]),), false, (γ = GammaShapeRate(2.0, 4.0),)),
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
    ("MvNMSP:(out,μ):gamma-γ", MvNormalMeanScalePrecision, (:out, :μ), (m = (out = MvNormalWeightedMeanPrecision([1.0, 2.0], I2), μ = MvNormalMeanCovariance([3.0, 4.0], [2.0 0.5; 0.5 1.0])), q = (γ = GammaShapeRate(2.0, 3.0),)), false),
    ("MvNMSP:(out,μ):point-mass-out", MvNormalMeanScalePrecision, (:out, :μ), (m = (out = PointMass([1.0, 1.0]), μ = MvNormalWeightedMeanPrecision([3.0, 4.0], I2)), q = (γ = GammaShapeRate(2.0, 3.0),)), false),
    ("MvNMSP:(out,μ):point-mass-μ", MvNormalMeanScalePrecision, (:out, :μ), (m = (out = MvNormalWeightedMeanPrecision([3.0, 4.0], I2), μ = PointMass([1.0, 1.0])), q = (γ = PointMass(0.5),)), false),
    ("MvNMSP:(out,μ,γ):point-mass-μ", MvNormalMeanScalePrecision, (:out, :μ, :γ), (m = (out = MvNormalWeightedMeanPrecision([1.0, 2.0], I2), μ = PointMass([1.0, 1.0]), γ = PointMass(0.5)),), false),
    ("MvNMSP:(out,μ,γ):point-mass-out", MvNormalMeanScalePrecision, (:out, :μ, :γ), (m = (out = PointMass([1.0, 1.0]), μ = MvNormalWeightedMeanPrecision([3.0, 4.0], I2), γ = PointMass(0.5)),), false),
    ("MvNMSP:(out,μ,γ):normals", MvNormalMeanScalePrecision, (:out, :μ, :γ), (m = (out = MvNormalWeightedMeanPrecision([1.0, 2.0], I2), μ = MvNormalMeanCovariance([3.0, 4.0], [2.0 0.5; 0.5 1.0]), γ = PointMass(0.5)),), false),
    ("MvNMSMP:(out,μ):gamma-γ", MvNormalMeanScaleMatrixPrecision, (:out, :μ), (m = (out = MvNormalWeightedMeanPrecision([1.0, 2.0], I2), μ = MvNormalMeanCovariance([3.0, 4.0], [2.0 0.5; 0.5 1.0])), q = (γ = GammaShapeRate(2.0, 3.0), G = Wishart(4.0, [3.0 0.5; 0.5 2.0]))), false),
    ("MvNMSMP:(out,μ):point-mass-out", MvNormalMeanScaleMatrixPrecision, (:out, :μ), (m = (out = PointMass([1.0, 1.0]), μ = MvNormalWeightedMeanPrecision([3.0, 4.0], I2)), q = (γ = GammaShapeRate(2.0, 3.0), G = Wishart(4.0, [3.0 0.5; 0.5 2.0]))), false),
    ("MvNMSMP:(out,μ):point-mass-μ", MvNormalMeanScaleMatrixPrecision, (:out, :μ), (m = (out = MvNormalWeightedMeanPrecision([3.0, 4.0], I2), μ = PointMass([1.0, 1.0])), q = (γ = PointMass(0.5), G = PointMass([2.0 0.5; 0.5 1.0]))), false),
    ("MvNMSMP:(out,μ,γ,G):point-mass-μ", MvNormalMeanScaleMatrixPrecision, (:out, :μ, :γ, :G), (m = (out = MvNormalWeightedMeanPrecision([1.0, 2.0], I2), μ = PointMass([1.0, 1.0]), γ = PointMass(0.5), G = PointMass([2.0 0.5; 0.5 1.0])),), false),
    ("MvNMSMP:(out,μ,γ,G):point-mass-out", MvNormalMeanScaleMatrixPrecision, (:out, :μ, :γ, :G), (m = (out = PointMass([1.0, 1.0]), μ = MvNormalWeightedMeanPrecision([3.0, 4.0], I2), γ = PointMass(0.5), G = PointMass([2.0 0.5; 0.5 1.0])),), false),
    ("MvNMSMP:(out,μ,γ,G):normals", MvNormalMeanScaleMatrixPrecision, (:out, :μ, :γ, :G), (m = (out = MvNormalWeightedMeanPrecision([1.0, 2.0], I2), μ = MvNormalMeanCovariance([3.0, 4.0], [2.0 0.5; 0.5 1.0]), γ = PointMass(0.5), G = PointMass([2.0 0.5; 0.5 1.0])),), false),
    ("Wishart:(out,ν,S):point-masses", Wishart, (:out, :ν, :S), (m = (out = Wishart(3.0, [1.0 0.2; 0.2 0.5]), ν = PointMass(3.0), S = PointMass([2.0 0.3; 0.3 1.5])),), false),
    ("InverseWishart:(out,ν,S):point-masses", InverseWishart, (:out, :ν, :S), (m = (out = ExponentialFamily.InverseWishartFast(4.0, [1.0 0.2; 0.2 0.5]), ν = PointMass(5.0), S = PointMass([2.0 0.3; 0.3 1.5])),), false),
    ("DirichletCollection:(out,a):point-mass", DirichletCollection, (:out, :a), (m = (out = DirichletCollection([2.0 1.0; 1.5 3.0; 1.0 2.5]), a = PointMass([1.0 2.0; 3.0 0.5; 2.0 1.5])),), false),
    ("MatrixNormal:(out,M,U,V):point-masses", MatrixNormal, (:out, :M, :U, :V), (m = (out = PointMass([1.0 2.0; 3.0 4.0; 5.0 6.0]), M = PointMass([0.5 1.0; 2.0 3.0; 4.0 5.0]), U = PointMass([2.0 0.3 0.0; 0.3 3.0 0.1; 0.0 0.1 4.0]), V = PointMass([1.0 0.5; 0.5 2.0])),), false),
    ("+:(in1,in2):point-mass-in2", +, (:in1, :in2), (m = (out = NormalMeanVariance(1.0, 2.0), in1 = NormalMeanPrecision(3.0, 0.5), in2 = PointMass(2.0)),), false),
    ("+:(in1,in2):point-mass-in1", +, (:in1, :in2), (m = (out = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.3; 0.3 1.0]), in1 = PointMass([1.0, 2.0]), in2 = MvNormalMeanCovariance([1.0, -1.0], [1.0 0.2; 0.2 2.0])),), false),
    ("+:(in1,in2):normals", +, (:in1, :in2), (m = (out = NormalMeanVariance(1.0, 2.0), in1 = NormalMeanPrecision(3.0, 0.5), in2 = NormalMeanVariance(0.0, 1.0)),), false),
    ("+:(in1,in2):mv-normals", +, (:in1, :in2), (m = (out = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.3; 0.3 1.0]), in1 = MvNormalMeanCovariance([1.0, -1.0], [1.0 0.2; 0.2 2.0]), in2 = MvNormalWeightedMeanPrecision([1.0, 1.0], [1.0 0.0; 0.0 1.5])),), false),
    ("-:(in1,in2):point-mass-in2", -, (:in1, :in2), (m = (out = NormalMeanVariance(1.0, 2.0), in1 = NormalMeanPrecision(3.0, 0.5), in2 = PointMass(2.0)),), false),
    ("-:(in1,in2):point-mass-in1", -, (:in1, :in2), (m = (out = NormalMeanVariance(1.0, 2.0), in1 = PointMass(2.0), in2 = NormalMeanPrecision(3.0, 0.5)),), SUBTRACTION_MARGINAL),
    ("-:(in1,in2):mv-point-mass-in1", -, (:in1, :in2), (m = (out = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.3; 0.3 1.0]), in1 = PointMass([1.0, 2.0]), in2 = MvNormalMeanCovariance([1.0, -1.0], [1.0 0.2; 0.2 2.0])),), SUBTRACTION_MARGINAL),
    ("-:(in1,in2):normals", -, (:in1, :in2), (m = (out = NormalMeanVariance(1.0, 2.0), in1 = NormalMeanPrecision(3.0, 0.5), in2 = NormalMeanVariance(0.0, 1.0)),), false),
    ("-:(in1,in2):mv-normals", -, (:in1, :in2), (m = (out = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.3; 0.3 1.0]), in1 = MvNormalMeanCovariance([1.0, -1.0], [1.0 0.2; 0.2 2.0]), in2 = MvNormalWeightedMeanPrecision([1.0, 1.0], [1.0 0.0; 0.0 1.5])),), false),
    ("dot:(in1,in2):point-mass-in1", dot, (:in1, :in2), (m = (out = NormalMeanVariance(1.0, 2.0), in1 = PointMass([2.0, 0.5]), in2 = MvNormalMeanCovariance([1.0, -1.0], [1.0 0.2; 0.2 2.0])),), false),
    ("dot:(in1,in2):point-mass-in2", dot, (:in1, :in2), (m = (out = NormalMeanVariance(1.0, 2.0), in1 = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.3; 0.3 1.0]), in2 = PointMass([2.0, 0.5])),), false),
    ("*:(A,in):point-mass-A", *, (:A, :in), (m = (out = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.3; 0.3 1.0]), A = PointMass([1.0 2.0; 0.5 2.0]), in = MvNormalMeanCovariance([1.0, -1.0], [1.0 0.2; 0.2 2.0])),), false),
    ("*:(A,in):point-mass-in", *, (:A, :in), (m = (out = NormalMeanVariance(1.0, 2.0), A = NormalMeanPrecision(1.0, 2.0), in = PointMass(2.0)),), false),
    ("*:(A,in):mv-precision-vector", *, (:A, :in), (m = (out = MvNormalMeanPrecision([1.0, 2.0], [3.0 0.5; 0.5 2.0]), A = NormalMeanPrecision(1.0, 2.0), in = PointMass([2.0, 0.5])),), false),
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
    ("MvNMSP:energy", MvNormalMeanScalePrecision, (q = (out = MvNormalMeanCovariance([1.5, 0.5], [1.0 0.2; 0.2 0.5]), μ = MvNormalMeanCovariance([1.0, 1.0], I2), γ = GammaShapeRate(2.0, 4.0)),), false),
    ("MvNMSP:energy:joint", MvNormalMeanScalePrecision, (q = (γ = GammaShapeRate(2.0, 4.0),), clusters = ((:out, :μ) => MvNormalMeanCovariance([1.0, 0.5, 2.0, 1.0], [2.0 0.1 0.2 0.0; 0.1 1.5 0.0 0.3; 0.2 0.0 1.0 0.1; 0.0 0.3 0.1 2.5]),)), false),
    ("MvNMSMP:energy", MvNormalMeanScaleMatrixPrecision, (q = (out = MvNormalMeanCovariance([1.5, 0.5], [1.0 0.2; 0.2 0.5]), μ = MvNormalMeanCovariance([1.0, 1.0], I2), γ = GammaShapeRate(2.0, 4.0), G = Wishart(4.0, [3.0 0.5; 0.5 2.0])),), false),
    ("MvNMSMP:energy:joint", MvNormalMeanScaleMatrixPrecision, (q = (γ = GammaShapeRate(2.0, 4.0), G = Wishart(4.0, [3.0 0.5; 0.5 2.0])), clusters = ((:out, :μ) => MvNormalMeanCovariance([1.0, 0.5, 2.0, 1.0], [2.0 0.1 0.2 0.0; 0.1 1.5 0.0 0.3; 0.2 0.0 1.0 0.1; 0.0 0.3 0.1 2.5]),)), false),
    ("Wishart:energy", Wishart, (q = (out = WISHART, ν = PointMass(3.0), S = PointMass([2.0 0.3; 0.3 1.5])),), false),
    ("Wishart:energy:inverse-wishart-S", Wishart, (q = (out = WISHART, ν = PointMass(3.0), S = INVERSE_WISHART),), false),
    ("InverseWishart:energy", InverseWishart, (q = (out = INVERSE_WISHART, ν = PointMass(5.0), S = PointMass([2.0 0.3; 0.3 1.5])),), false),
    ("InverseWishart:energy:wishart-S", InverseWishart, (q = (out = INVERSE_WISHART, ν = PointMass(5.0), S = WISHART),), false),
    ("DirichletCollection:energy", DirichletCollection, (q = (out = DirichletCollection([2.0 1.0; 1.5 3.0; 1.0 2.5]), a = PointMass([1.0 2.0; 3.0 0.5; 2.0 1.5])),), false),
    ("DirichletCollection:energy:rank-3", DirichletCollection, (q = (out = DirichletCollection(reshape(collect(1.0:12.0) ./ 4, 3, 2, 2)), a = PointMass(reshape(collect(12.0:-1.0:1.0) ./ 5, 3, 2, 2))),), false),
    ("MvNormalGamma:energy", MvNormalGamma, (q = (out = MvNormalGamma([0.3, 0.1], [1.5 0.2; 0.2 1.0], 4.2, 2.5), μ = PointMass([0.5, -1.0]), Λ = PointMass([2.0 0.3; 0.3 1.5]), α = PointMass(2.0), β = PointMass(3.0)),), false),
    ("MatrixNormal:energy:point-masses", MatrixNormal, (q = (out = PointMass([1.0 2.0; 3.0 4.0; 5.0 6.0]), M = PointMass([0.5 1.0; 2.0 3.0; 4.0 5.0]), U = PointMass([2.0 0.3 0.0; 0.3 3.0 0.1; 0.0 0.1 4.0]), V = PointMass([1.0 0.5; 0.5 2.0])),), false),
    ("MatrixNormal:energy", MatrixNormal, (q = (out = MatrixNormal([0.5 1.0; 2.0 3.0; 4.0 5.0], [0.5 0.1 0.0; 0.1 0.5 0.0; 0.0 0.0 0.5], [1.0 0.2; 0.2 0.5]), M = MatrixNormal([0.5 1.0; 2.0 3.0; 4.0 5.0], [0.5 0.1 0.0; 0.1 0.5 0.0; 0.0 0.0 0.5], [1.0 0.2; 0.2 0.5]), U = InverseWishart(7.0, [2.0 0.3 0.0; 0.3 3.0 0.1; 0.0 0.1 4.0]), V = InverseWishart(6.0, [1.0 0.5; 0.5 2.0])),), false),
    ("MatrixNormalWishart:energy", MatrixNormalWishart, (q = (out = MatrixNormalWishart([0.5 1.0; 1.5 2.0], [1.5 0.2; 0.2 1.0], [1.0 0.1; 0.1 1.2], 6.0), M = PointMass([1.0 2.0; 3.0 4.0]), U = PointMass([2.0 0.3; 0.3 1.5]), V = PointMass([1.0 0.2; 0.2 1.5]), ν = PointMass(5.0)),), false),
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
# The multivariate form: Wishart precisions and a two-dimensional `out`.
const MV_MIXTURE_COMPONENTS = (
    m = (MvNormalMeanCovariance([0.0, 1.0], [1.0 0.2; 0.2 2.0]), MvNormalWeightedMeanPrecision([2.0, 1.0], [1.5 0.0; 0.0 1.0])),
    p = (Wishart(3.0, [1.0 0.1; 0.1 0.5]), Wishart(4.0, [0.5 0.0; 0.0 1.0])),
)
const MV_MIXTURE_OUT = MvNormalMeanCovariance([1.5, 0.5], [0.5 0.1; 0.1 0.3])

const MIXTURE_CASES = [
    ("NormalMixture:(m,1)", (:m, 1), (q = (out = PointMass(1.5), switch = MIXTURE_SWITCH, p = (MIXTURE_COMPONENTS.p[1], nothing)),), (q = (out = PointMass(1.5), switch = MIXTURE_SWITCH, p = MIXTURE_COMPONENTS.p[1]),)),
    ("NormalMixture:(m,2)", (:m, 2), (q = (out = NormalMeanVariance(1.0, 0.5), switch = MIXTURE_SWITCH, p = (nothing, MIXTURE_COMPONENTS.p[2])),), (q = (out = NormalMeanVariance(1.0, 0.5), switch = MIXTURE_SWITCH, p = MIXTURE_COMPONENTS.p[2]),)),
    ("NormalMixture:(p,1)", (:p, 1), (q = (out = PointMass(1.5), switch = MIXTURE_SWITCH, m = (MIXTURE_COMPONENTS.m[1], nothing)),), (q = (out = PointMass(1.5), switch = MIXTURE_SWITCH, m = MIXTURE_COMPONENTS.m[1]),)),
    ("NormalMixture:(p,2)", (:p, 2), (q = (out = PointMass(1.5), switch = MIXTURE_SWITCH, m = (nothing, MIXTURE_COMPONENTS.m[2])),), (q = (out = PointMass(1.5), switch = MIXTURE_SWITCH, m = MIXTURE_COMPONENTS.m[2]),)),
    ("NormalMixture:switch", :switch, (q = (out = PointMass(1.5), MIXTURE_COMPONENTS...),), (q = (out = PointMass(1.5), MIXTURE_COMPONENTS...),)),
    ("NormalMixture:out", :out, (q = (switch = MIXTURE_SWITCH, MIXTURE_COMPONENTS...),), (q = (switch = MIXTURE_SWITCH, MIXTURE_COMPONENTS...),)),
    ("NormalMixture:mv:(m,1)", (:m, 1), (q = (out = MV_MIXTURE_OUT, switch = MIXTURE_SWITCH, p = (MV_MIXTURE_COMPONENTS.p[1], nothing)),), (q = (out = MV_MIXTURE_OUT, switch = MIXTURE_SWITCH, p = MV_MIXTURE_COMPONENTS.p[1]),)),
    ("NormalMixture:mv:(m,2)", (:m, 2), (q = (out = PointMass([1.0, 2.0]), switch = MIXTURE_SWITCH, p = (nothing, MV_MIXTURE_COMPONENTS.p[2])),), (q = (out = PointMass([1.0, 2.0]), switch = MIXTURE_SWITCH, p = MV_MIXTURE_COMPONENTS.p[2]),)),
    ("NormalMixture:mv:(p,1)", (:p, 1), (q = (out = MV_MIXTURE_OUT, switch = MIXTURE_SWITCH, m = (MV_MIXTURE_COMPONENTS.m[1], nothing)),), (q = (out = MV_MIXTURE_OUT, switch = MIXTURE_SWITCH, m = MV_MIXTURE_COMPONENTS.m[1]),)),
    ("NormalMixture:mv:(p,2)", (:p, 2), (q = (out = PointMass([1.0, 2.0]), switch = MIXTURE_SWITCH, m = (nothing, MV_MIXTURE_COMPONENTS.m[2])),), (q = (out = PointMass([1.0, 2.0]), switch = MIXTURE_SWITCH, m = MV_MIXTURE_COMPONENTS.m[2]),)),
    ("NormalMixture:mv:switch", :switch, (q = (out = MV_MIXTURE_OUT, MV_MIXTURE_COMPONENTS...),), (q = (out = MV_MIXTURE_OUT, MV_MIXTURE_COMPONENTS...),)),
    ("NormalMixture:mv:out", :out, (q = (switch = MIXTURE_SWITCH, MV_MIXTURE_COMPONENTS...),), (q = (switch = MIXTURE_SWITCH, MV_MIXTURE_COMPONENTS...),)),
]

const V6_GAMMA_MIXTURE = V6Oracle.ReactiveMP.GammaMixture{2}
const GAMMA_COMPONENTS = (a = (GammaShapeRate(2.0, 3.0), PointMass(4.0)), b = (GammaShapeRate(1.5, 2.5), GammaShapeRate(3.5, 4.5)))
const GAMMA_SWITCH = Categorical([0.3, 0.7])
const GAMMA_MIXTURE_CASES = [
    ("GammaMixture:(a,1)", (:a, 1), (q = (out = GammaShapeRate(2.0, 1.0), switch = GAMMA_SWITCH, b = (GAMMA_COMPONENTS.b[1], nothing)),), (q = (out = GammaShapeRate(2.0, 1.0), switch = GAMMA_SWITCH, b = GAMMA_COMPONENTS.b[1]),)),
    ("GammaMixture:(b,2)", (:b, 2), (q = (out = GammaShapeRate(2.0, 1.0), switch = GAMMA_SWITCH, a = (nothing, GAMMA_COMPONENTS.a[2])),), (q = (out = GammaShapeRate(2.0, 1.0), switch = GAMMA_SWITCH, a = GAMMA_COMPONENTS.a[2]),)),
    ("GammaMixture:out", :out, (q = (switch = GAMMA_SWITCH, GAMMA_COMPONENTS...),), (q = (switch = GAMMA_SWITCH, GAMMA_COMPONENTS...),)),
    ("GammaMixture:switch", :switch, (q = (out = GammaShapeRate(2.0, 1.0), GAMMA_COMPONENTS...),), (q = (out = GammaShapeRate(2.0, 1.0), GAMMA_COMPONENTS...),)),
]

# `flagged` is `false`, `true` for #669, or the reason of another declared correction.
declared_reason(flagged::Bool) = NMV_669
declared_reason(flagged::AbstractString) = flagged
declare(id, flagged) = flagged === false ? DeclaredDisagreement[] : [DeclaredDisagreement(id; kind = :correction, reasoning = declared_reason(flagged))]

@testset "StandardMessagePassingRules against v6" begin
    @testset "message rules" begin
        for (id, node, edge, inputs, flagged) in MESSAGE_CASES
            m, q = get(inputs, :m, NamedTuple()), get(inputs, :q, NamedTuple())
            result = call_message_update_rule(node, edge; m, q)
            v7, v7_logscale = getresult(result), getlogscale(result)
            v6, v6_logscale = v6_message_update(v6_node(node), edge, m, q)
            # Where v6 has a log scale, v7's must agree; v7 also declares some v6 left out.
            actual_logscale = v6_logscale === nothing ? nothing : (isdefined_logscale(v7_logscale) ? v7_logscale : nothing)
            record = compare_with_reference(id, v7, as_v7(node, v6); inputs, node = string(node), target = ":$edge", actual_logscale, reference_logscale = v6_logscale, declared = declare(id, flagged))
            # A declared correction must actually differ, or the declaration is stale.
            @test (flagged !== false) == (record.outcome === :correction)
        end
    end
    @testset "message rules consuming a joint" begin
        for (id, node, edge, clusters, flagged, q...) in CLUSTER_MESSAGE_CASES
            q = isempty(q) ? NamedTuple() : only(q)
            v7 = getresult(call_message_update_rule(node, edge; q, clusters))
            v6, _ = v6_message_update(node, edge, NamedTuple(), merge(NamedTuple{map(V6Oracle.v6_name, Tuple(first.(clusters)))}(Tuple(last.(clusters))), q))
            record = compare_with_reference(id, v7, v6; inputs = clusters, node = string(node), target = ":$edge", declared = declare(id, flagged))
            @test (flagged !== false) == (record.outcome === :correction)
        end
    end
    @testset "NormalMixture" begin
        for (id, target, v7_inputs, v6_inputs) in MIXTURE_CASES
            v7 = getresult(call_message_update_rule(NormalMixture, target; q = v7_inputs.q))
            v6, _ = v6_message_update(V6_NORMAL_MIXTURE, target, NamedTuple(), v6_inputs.q)
            @test compare_with_reference(id, v7, v6; inputs = v7_inputs, node = "NormalMixture", target = string(target)).outcome === :agree
        end
        for (id, q) in [
                ("NormalMixture:energy", (out = PointMass(1.5), switch = MIXTURE_SWITCH, MIXTURE_COMPONENTS...)),
                ("NormalMixture:mv:energy", (out = MV_MIXTURE_OUT, switch = MIXTURE_SWITCH, MV_MIXTURE_COMPONENTS...)),
            ]
            v7 = getresult(call_average_energy(NormalMixture; q))
            v6 = v6_average_energy(V6_NORMAL_MIXTURE, q)
            @test compare_with_reference(id, v7, v6; node = "NormalMixture", target = "energy").outcome === :agree
        end
    end
    @testset "GammaMixture" begin
        for (id, target, v7_inputs, v6_inputs) in GAMMA_MIXTURE_CASES
            v7 = getresult(call_message_update_rule(GammaMixture, target; q = v7_inputs.q))
            v6, _ = v6_message_update(V6_GAMMA_MIXTURE, target, NamedTuple(), v6_inputs.q)
            @test compare_with_reference(id, v7, as_v7(GammaMixture, v6); inputs = v7_inputs, node = "GammaMixture", target = string(target)).outcome === :agree
        end
        # v6's energy takes GammaShapeRate rates only.
        q = (out = GammaShapeRate(2.0, 1.0), switch = GAMMA_SWITCH, GAMMA_COMPONENTS...)
        @test compare_with_reference("GammaMixture:energy", getresult(call_average_energy(GammaMixture; q)), v6_average_energy(V6_GAMMA_MIXTURE, q); node = "GammaMixture", target = "energy").outcome === :agree
    end
    # Belief propagation towards NMV's `v` is a log-density on the half line with no family;
    # the port and v6 are compared by evaluating it.
    @testset "log-density messages" begin
        for m in [
                (out = PointMass(2.0), μ = NormalMeanVariance(0.0, 1.0)),
                (out = NormalMeanVariance(0.5, 2.0), μ = PointMass(-3.5)),
                (out = NormalMeanVariance(1.0, 0.5), μ = NormalMeanVariance(-1.0, 2.0)),
            ]
            v7 = getresult(call_message_update_rule(NormalMeanVariance, :v; m))
            v6, _ = v6_message_update(NormalMeanVariance, :v, m, NamedTuple())
            @test all(v -> logpdf(v7, v) ≈ logpdf(v6, v), (0.1, 1.0, 3.5, 10.0))
        end
    end
    # `*`'s messages without a closed form are log-densities: both versions are evaluated against
    # a midpoint rule over their defining integrals, whose integrands are bounded here. v6's
    # sampled rules draw from its global generator, seeded; the port's from a seeded Xoshiro.
    # The tolerances hold for any stream. v6's sampled message towards a factor weights each
    # draw by |y|, the density of out/y, so only it misses its integral.
    @testset "* log-density messages" begin
        integral(f, lo, hi) = (h = (hi - lo) / 100_000; h * sum(k -> f(lo + (k - 1 / 2) * h), 1:100_000))
        m_out, m_y = NormalMeanVariance(1.5, 0.5), NormalMeanVariance(0.8, 0.3)
        v7 = getresult(call_message_update_rule(*, :in; m = (out = m_out, A = m_y)))
        v6, _ = v6_message_update(*, :in, (out = m_out, A = m_y), NamedTuple())
        for x in (-2.0, 0.7, 1.9)
            @test logpdf(v7, x) ≈ logpdf(v6, x)
        end
        m_A, m_in = NormalMeanVariance(1.0, 0.5), NormalMeanVariance(0.5, 0.4)
        v7 = getresult(call_message_update_rule(*, :out; m = (A = m_A, in = m_in)))
        v6, _ = v6_message_update(*, :out, (A = m_A, in = m_in), NamedTuple())
        for z in (-1.0, 0.3, 2.5)
            @test logpdf(v7, z) ≈ logpdf(v6, z)
        end
        g, b = GammaShapeRate(3.0, 2.0), Beta(2.0, 3.0)
        Random.seed!(42)
        v6_in, _ = v6_message_update(*, :in, (out = g, A = b), NamedTuple())
        v7_in = getresult(call_message_update_rule(*, :in; m = (out = g, A = b), ctx = RuleContext(rng = Random.Xoshiro(42))))
        for x in (0.5, 2.0, 4.0)
            exact = log(integral(y -> pdf(g, x * y) * pdf(b, y), 0.0, 1.0)) + log(3000)
            @test logpdf(v7_in, x) ≈ exact atol = 0.05
            @test !isapprox(logpdf(v6_in, x), exact; atol = 0.3)
        end
        @info "known disagreement `*:in:sampled` (correction): ReactiveMP.jl#679: v6 weights each draw of the other factor by |y|, the density of out/y; the message is ∫ p_out(x y) p_y(y) dy"
        v6_out, _ = v6_message_update(*, :out, (A = b, in = g), NamedTuple())
        v7_out = getresult(call_message_update_rule(*, :out; m = (A = b, in = g), ctx = RuleContext(rng = Random.Xoshiro(42))))
        for z in (0.3, 1.0, 2.0)
            exact = log(integral(a -> pdf(b, a) * pdf(g, z / a) / a, 0.0, 1.0)) + log(3000)
            @test logpdf(v7_out, z) ≈ exact atol = 0.05
            @test logpdf(v6_out, z) ≈ exact atol = 0.05
        end
    end
    @testset "marginal rules" begin
        for (id, node, members, inputs, flagged) in MARGINAL_CASES
            m, q = get(inputs, :m, NamedTuple()), get(inputs, :q, NamedTuple())
            v7 = getresult(call_marginal_update_rule(node, members; m, q))
            v6 = v6_marginal_update(v6_node(node), members, m, q)
            record = compare_with_reference(id, v7, as_v7(node, v6); inputs, node = string(node), target = string(members), declared = declare(id, flagged))
            @test (flagged !== false) == (record.outcome === :correction)
        end
    end
    @testset "average energies" begin
        for (id, node, inputs, flagged) in AVERAGE_ENERGY_CASES
            q, clusters = get(inputs, :q, NamedTuple()), get(inputs, :clusters, ())
            v7 = getresult(call_average_energy(node; q, clusters))
            v6 = v6_average_energy(v6_node(node), q, clusters)
            record = compare_with_reference(id, v7, v6; inputs, node = string(node), target = "energy", declared = declare(id, flagged))
            @test (flagged !== false) == (record.outcome === :correction)
        end
    end
end
