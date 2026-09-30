# `out = in[1] + in[2] + …`: the function `+` is the node, with any number of summands from two,
# so `a + b + c`, one call of `+`, is one node with no intermediate sum.
@define_factor_node(node = +, type = Deterministic, interfaces = [:out, :in...], min_group_length = 2)

# A Gaussian message or a known value; the arithmetic nodes' rules take either.
const NormalOrPoint = Union{PointMass, NormalDistributionsFamily}

# The message for `a + b` and for `a - b`, given the messages of `a` and `b`, in the parameters
# the inputs come in. `+` and `-` both use them: towards `out` of `+` is `sum_message(in1, in2)`,
# towards `in1` is `difference_message(out, in2)`, and so on. A multivariate normal's moments are
# taken together, with `mean_cov`.
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

# The sum of a tuple of messages, skipping the `nothing` a rule's `m[:in][!k]` holds at the
# target's own position. Pairwise, left to right, so two messages give `sum_message` itself.
sum_of_messages(messages::Tuple) = foldl(add_message, messages)
add_message(a, b) = sum_message(a, b)
add_message(::Nothing, b) = b
add_message(a, ::Nothing) = a
add_message(::Nothing, ::Nothing) = nothing

# The joint of Gaussian messages `members` times a Gaussian factor of their sum with weighted mean
# ξ and precision W: each member's (ξᵢ, Wᵢ) plus ξ, and W added to every block of the precision,
# the members stacked in order. One member gives its own normal, univariate or multivariate.
function gaussian_sum_joint(ξ, W, members::Tuple)
    parts = map(weightedmean_precision, members)
    length(parts) == 1 && return weighted_normal(first(first(parts)) + ξ, last(first(parts)) + W)
    return stacked_sum_joint(ξ, W, parts)
end

weighted_normal(ξ::Real, W::Real) = NormalWeightedMeanPrecision(ξ, W)
weighted_normal(ξ::AbstractVector, W::AbstractMatrix) = MvNormalWeightedMeanPrecision(ξ, W)

function stacked_sum_joint(ξ::Real, W::Real, parts)
    precision = fill(W, length(parts), length(parts))
    for (i, (_, Wᵢ)) in enumerate(parts)
        precision[i, i] += Wᵢ
    end
    return MvNormalWeightedMeanPrecision([ξᵢ + ξ for (ξᵢ, _) in parts], precision)
end

function stacked_sum_joint(ξ::AbstractVector, W::AbstractMatrix, parts)
    d, n = length(ξ), length(parts)
    precision = repeat(W, n, n)
    for (i, (_, Wᵢ)) in enumerate(parts)
        block = ((i - 1) * d + 1):(i * d)
        precision[block, block] .+= Wᵢ
    end
    return MvNormalWeightedMeanPrecision(reduce(vcat, [ξᵢ .+ ξ for (ξᵢ, _) in parts]), precision)
end

# The positions of the Gaussian members and the sum of the point masses among `inputs`.
gaussian_members(inputs::Tuple) = Tuple(k for k in eachindex(inputs) if !(inputs[k] isa PointMass))
point_sum(inputs::Tuple) = sum(mean(input) for input in inputs if input isa PointMass; init = zero(mean(first(inputs))))

# The joint of the inputs of `+` given a Gaussian message on `out`: the Gaussian members jointly,
# their messages times `out`'s message of their sum shifted by the point masses, each point mass
# a block of its own.
function sum_inputs_joint(m_out::NormalDistributionsFamily, inputs::Tuple)
    gaussian = gaussian_members(inputs)
    ξ, W = weightedmean_precision(m_out)
    isempty(gaussian) || (ξ = ξ - W * point_sum(inputs))
    joint = isempty(gaussian) ? nothing : gaussian_sum_joint(ξ, W, map(k -> inputs[k], gaussian))
    length(gaussian) == length(inputs) && return joint
    blocks = Pair[((:in, k),) => inputs[k] for k in eachindex(inputs) if inputs[k] isa PointMass]
    isempty(gaussian) || push!(blocks, Tuple((:in, k) for k in gaussian) => joint)
    sort!(blocks; by = block -> last(first(first(block))))
    return promote_cluster(FactorizedCluster(blocks...), m_out, inputs...)
end

"""
    StandardMessagePassingRules.InputsGivenSum

The joint of the inputs of `+` when their sum is known, the marginal rule's result for a
`PointMass` on `out`: the inputs then lie on the plane where they sum to it. The point-mass
inputs are known, and of the Gaussian ones every member but the last is free; the last is their
difference from the sum. Its entropy is the free members' joint entropy and one point-mass
entropy for each known input and for the sum, which the free energy needs.

# Fields

- `free`: the joint of the free Gaussian members, their messages times the last member's message
  of the difference, or `nothing` when at most one input is Gaussian;
- `points`: the point-mass inputs, a tuple;
- `sum`: the sum, the `PointMass` on `out`.
"""
struct InputsGivenSum{F, P, S}
    free::F
    points::P
    sum::S
end

function BayesBase.entropy(joint::InputsGivenSum)
    known = mapreduce(BayesBase.entropy, +, joint.points; init = BayesBase.entropy(joint.sum))
    return joint.free === nothing ? known : BayesBase.entropy(joint.free) + known
end
BayesBase.paramfloattype(joint::InputsGivenSum) = BayesBase.paramfloattype(joint.sum)
BayesBase.convert_paramfloattype(::Type{T}, joint::InputsGivenSum) where {T} = InputsGivenSum(
    joint.free === nothing ? nothing : BayesBase.convert_paramfloattype(T, joint.free),
    map(point -> BayesBase.convert_paramfloattype(T, point), joint.points),
    BayesBase.convert_paramfloattype(T, joint.sum),
)

# Given the sum c, the last Gaussian member is c' - Σ of the others, c' the sum less the point
# masses: its message (ξ, W) of that difference is a Gaussian factor of the others' sum with
# weighted mean W c' - ξ and precision W.
function sum_inputs_joint(m_out::PointMass, inputs::Tuple)
    gaussian = gaussian_members(inputs)
    points = Tuple(input for input in inputs if input isa PointMass)
    T = BayesBase.promote_paramfloattype(m_out, inputs...)
    free = if length(gaussian) <= 1
        nothing
    else
        ξ, W = weightedmean_precision(inputs[last(gaussian)])
        BayesBase.convert_paramfloattype(T, gaussian_sum_joint(W * (mean(m_out) - point_sum(inputs)) - ξ, W, map(k -> inputs[k], Base.front(gaussian))))
    end
    return InputsGivenSum(free, map(point -> BayesBase.convert_paramfloattype(T, point), points), BayesBase.convert_paramfloattype(T, m_out))
end

# The joint of two Gaussian inputs of `out = in1 + s·in2`, s = ±1: their messages times
# m_out(in1 + s·in2), which adds [W W s; W s W] to the precision and [ξ; s ξ] to the weighted
# mean, with (ξ, W) the output message's.
function input_joint(m_out, m_in1, m_in2, s)
    ξ, W = weightedmean_precision(m_out)
    ξ1, W1 = weightedmean_precision(m_in1)
    ξ2, W2 = weightedmean_precision(m_in2)
    return MvNormalWeightedMeanPrecision([ξ1 .+ ξ; ξ2 .+ s .* ξ], [W1 .+ W s .* W; s .* W W2 .+ W])
end
