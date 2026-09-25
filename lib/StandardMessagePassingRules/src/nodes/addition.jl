# `out = in1 + in2`: the function `+` is the node, as in v6.
@define_factor_node(node = +, type = Deterministic, interfaces = [:out, :in1, :in2])

# A Gaussian message or a known value; the arithmetic nodes' rules take either.
const NormalOrPoint = Union{PointMass, NormalDistributionsFamily}

# The message for `a + b` and for `a - b`, given the messages of `a` and `b`, in the parameters
# the inputs come in. `+` and `-` both use them: towards `out` of `+` is `sum_message(in1, in2)`,
# towards `in1` is `difference_message(out, in2)`, and so on. A multivariate normal's moments are
# taken together, `mean_cov`, as v6 took them, which rounds as v6 did.
sum_message(a::PointMass, b::PointMass) = PointMass(mean(a) + mean(b))
sum_message(a::Distribution, b::Distribution) = convolve(a, b)
sum_message(a::UnivariateNormalDistributionsFamily, b::UnivariateNormalDistributionsFamily) = NormalMeanVariance(mean(a) + mean(b), var(a) + var(b))
sum_message(a::MultivariateNormalDistributionsFamily, b::MultivariateNormalDistributionsFamily) = ((μa, Σa) = mean_cov(a); (μb, Σb) = mean_cov(b); MvNormalMeanCovariance(μa + μb, Σa + Σb))
sum_message(a::PointMass, b::NormalDistributionsFamily) = sum_message(b, a)
sum_message(a::UnivariateNormalDistributionsFamily, b::PointMass) = NormalMeanVariance(mean(a) + mean(b), var(a))
sum_message(a::MultivariateNormalDistributionsFamily, b::PointMass) = ((μ, Σ) = mean_cov(a); MvNormalMeanCovariance(μ + mean(b), Σ))
sum_message(a::NormalMeanPrecision, b::PointMass) = NormalMeanPrecision(mean(a) + mean(b), precision(a))
sum_message(a::MvNormalMeanPrecision, b::PointMass) = MvNormalMeanPrecision(mean(a) + mean(b), precision(a))
sum_message(a::MvNormalWeightedMeanPrecision, b::PointMass) = ((ξ, W) = weightedmean_precision(a); MvNormalWeightedMeanPrecision(ξ + W * mean(b), W))

difference_message(a::PointMass, b::PointMass) = PointMass(mean(a) - mean(b))
difference_message(a::UnivariateNormalDistributionsFamily, b::UnivariateNormalDistributionsFamily) = NormalMeanVariance(mean(a) - mean(b), var(a) + var(b))
difference_message(a::MultivariateNormalDistributionsFamily, b::MultivariateNormalDistributionsFamily) = ((μa, Σa) = mean_cov(a); (μb, Σb) = mean_cov(b); MvNormalMeanCovariance(μa - μb, Σa + Σb))
difference_message(a::UnivariateNormalDistributionsFamily, b::PointMass) = NormalMeanVariance(mean(a) - mean(b), var(a))
difference_message(a::PointMass, b::UnivariateNormalDistributionsFamily) = NormalMeanVariance(mean(a) - mean(b), var(b))
difference_message(a::MultivariateNormalDistributionsFamily, b::PointMass) = ((μ, Σ) = mean_cov(a); MvNormalMeanCovariance(μ - mean(b), Σ))
difference_message(a::PointMass, b::MultivariateNormalDistributionsFamily) = ((μ, Σ) = mean_cov(b); MvNormalMeanCovariance(mean(a) - μ, Σ))
difference_message(a::NormalMeanPrecision, b::PointMass) = NormalMeanPrecision(mean(a) - mean(b), precision(a))
difference_message(a::PointMass, b::NormalMeanPrecision) = NormalMeanPrecision(mean(a) - mean(b), precision(b))
difference_message(a::MvNormalMeanPrecision, b::PointMass) = MvNormalMeanPrecision(mean(a) - mean(b), precision(a))
difference_message(a::PointMass, b::MvNormalMeanPrecision) = MvNormalMeanPrecision(mean(a) - mean(b), precision(b))
difference_message(a::MvNormalWeightedMeanPrecision, b::PointMass) = ((ξ, W) = weightedmean_precision(a); MvNormalWeightedMeanPrecision(ξ - W * mean(b), W))
difference_message(a::PointMass, b::MvNormalWeightedMeanPrecision) = ((ξ, W) = weightedmean_precision(b); MvNormalWeightedMeanPrecision(W * mean(a) - ξ, W))

# The joint of two Gaussian inputs of `out = in1 + s·in2`, s = ±1: their messages times
# m_out(in1 + s·in2), which adds [W W s; W s W] to the precision and [ξ; s ξ] to the weighted
# mean, with (ξ, W) the output message's.
function input_joint(m_out, m_in1, m_in2, s)
    ξ, W = weightedmean_precision(m_out)
    ξ1, W1 = weightedmean_precision(m_in1)
    ξ2, W2 = weightedmean_precision(m_in2)
    return MvNormalWeightedMeanPrecision([ξ1 .+ ξ; ξ2 .+ s .* ξ], [W1 .+ W s .* W; s .* W W2 .+ W])
end
