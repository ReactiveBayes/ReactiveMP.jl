# The rules of `CVIProjection`, ported from v6's `ReactiveMPProjectionExt`: loaded with
# ExponentialFamilyProjection, which does the projections.
module DeltaMessagePassingRulesProjectionExt

using DeltaMessagePassingRules, MessagePassingRulesBase, ExponentialFamilyProjection, ExponentialFamily, BayesBase, Distributions
using DeltaMessagePassingRules: CVIProjection, FullSampling, MeanBased, get_kth_in_form
using MessagePassingRulesBase: Target, getnodefn

DeltaMessagePassingRules.is_delta_node_compatible(::CVIProjection) = Val(true)

# The message towards `out` is the projected marginal of `out` divided by the message that
# arrived on it, left lazy: its product with that message, at the variable, is the projection.
struct DivisionOf{A, B}
    numerator::A
    denumerator::B
end

(division::DivisionOf)(x) = logpdf(division, x)
BayesBase.insupport(d::DivisionOf, p) = insupport(d.numerator, p) && insupport(d.denumerator, p)
BayesBase.logpdf(d::DivisionOf, p) = logpdf(d.numerator, p) - logpdf(d.denumerator, p)

function BayesBase.prod(::GenericProd, something::DivisionOf, division::DivisionOf)
    division.denumerator == something.numerator && return DivisionOf(division.numerator, something.denumerator)
    division.numerator == something.denumerator && return DivisionOf(something.numerator, division.denumerator)
    return ProductOf(something, division)
end
BayesBase.prod(::GenericProd, something, division::DivisionOf) = prod(GenericProd(), division, something)
BayesBase.prod(::GenericProd, division::DivisionOf, something) =
    division.denumerator == something ? division.numerator : ProductOf(division, something)
BayesBase.prod(::GenericProd, division::DivisionOf, ::Missing) = division
BayesBase.prod(::GenericProd, ::Missing, division::DivisionOf) = division
BayesBase.prod(::GenericProd, product::ProductOf, division::DivisionOf) = ProductOf(product, division)

# Samples as a vector of points: a matrix of draws, one per column, becomes its columns.
cvilinearize(vector::AbstractVector) = vector
cvilinearize(matrix::AbstractMatrix) = eachcol(matrix)

# The projection family: the one the method names, or that of `reference` with the method's
# parameters, or the defaults.
function projection_to(reference, dims, parameters)
    T = ExponentialFamily.exponential_family_typetag(reference)
    conditioner = getconditioner(convert(ExponentialFamilyDistribution, reference))
    return ProjectedTo(T, dims...; conditioner, parameters)
end

create_project_to(::CVIProjection{S, Nothing}, q_out, samples) where {S} =
    projection_to(q_out, size(first(samples)), ExponentialFamilyProjection.DefaultProjectionParameters())
create_project_to(method::CVIProjection{S, <:ProjectedTo}, q_out, samples) where {S} = method.out_prjparams
create_project_to(method::CVIProjection{S, <:ProjectionParameters}, q_out, samples) where {S} =
    projection_to(q_out, size(first(samples)), method.out_prjparams)

create_project_to_ins(::CVIProjection, ::Nothing, m_in) =
    projection_to(m_in, size(m_in), ExponentialFamilyProjection.DefaultProjectionParameters())
create_project_to_ins(::CVIProjection, form::ProjectedTo, m_in) = form
create_project_to_ins(::CVIProjection, parameters::ProjectionParameters, m_in) = projection_to(m_in, size(m_in), parameters)
create_project_to_ins(method::CVIProjection, m_in, k::Int) = create_project_to_ins(method, get_kth_in_form(method, k), m_in)

generic_logpdf(reference, f) =
    convert(promote_variate_type(variate_form(typeof(reference)), BayesBase.AbstractContinuousGenericLogPdf), BayesBase.UnspecifiedDomain(), f)

# Samples of the inputs, as tuples: from the proposal once there is one, from the messages before.
input_samples(rng, ::Nothing, m_ins, strategy::FullSampling) = zip(map(m_in -> cvilinearize(rand(rng, m_in, strategy.samples)), m_ins)...)
input_samples(rng, ::Nothing, m_ins, ::MeanBased) = zip(map(m_in -> [mean(m_in)], m_ins)...)
input_samples(rng, proposal::FactorizedJoint, m_ins, strategy::FullSampling) =
    zip(map(q_in -> cvilinearize(rand(rng, q_in, strategy.samples)), components(proposal))...)
input_samples(rng, proposal::FactorizedJoint, m_ins, ::MeanBased) = zip(map(q_in -> [mean(q_in)], components(proposal))...)

replace_at(t::Tuple, i, z) = ntuple(j -> j == i ? z : t[j], length(t))

@define_message_update_rule(
    node = DeltaFn, target = :out, algorithm = DeltaApproximation{<:CVIProjection}, ctx = (:node, :rng),
    args = (m[:out]::Any, q[:out]::Any, q[(:in,)]::FactorizedJoint),
    body = (algo, ctx, args) -> begin
        f = getnodefn(ctx.node, Target(:out))
        method = algo.method
        samples = map(q_in -> cvilinearize(rand(ctx.rng, sampling_optimized(q_in), method.outsamples)), components(args.q[(:in,)]))
        out_samples = map(x -> f(x...), zip(samples...))
        estimate = project_to(create_project_to(method, args.q[:out], out_samples), out_samples)
        DivisionOf(estimate, args.m[:out])
    end,
)

@define_message_update_rule(
    node = DeltaFn, target = (:in, k), algorithm = DeltaApproximation{<:CVIProjection},
    args = (m[:in][k]::Any, q[(:in,)]::FactorizedJoint),
    body = (args) -> DivisionOf(component(args.q[(:in,)], k), args.m[:in][k]),
)

# The joint over the inputs, a product of their projections. One input is projected directly;
# several, each against samples of the others, and the result becomes the next proposal.
@define_marginal_update_rule(
    node = DeltaFn, target = (:in,), algorithm = DeltaApproximation{<:CVIProjection}, ctx = (:node, :rng), pure = false,
    args = (m[:out]::Any, m[:in...]::Any),
    body = (algo, ctx, args) -> begin
        g = getnodefn(ctx.node, Target(:out))
        method, m_out, m_ins = algo.method, args.m[:out], args.m[:in]
        if length(m_ins) == 1
            m_in = only(m_ins)
            logp = generic_logpdf(m_in, z -> logpdf(m_out, g(z)))
            FactorizedJoint((project_to(create_project_to_ins(method, m_in, 1), logp, m_in),))
        else
            proposal = method.proposal_distribution
            samples = input_samples(ctx.rng, proposal.distribution, m_ins, method.sampling_strategy)
            # The expected log-likelihood of `out` with input `i` at `z`, the others sampled.
            loglikelihood(z, i) = mean(sample -> logpdf(m_out, g(replace_at(sample, i, z)...)), samples)
            projections = ntuple(length(m_ins)) do i
                m_in = m_ins[i]
                prj = create_project_to_ins(method, m_in, i)
                matches = ExponentialFamilyProjection.get_projected_to_type(prj) === ExponentialFamily.exponential_family_typetag(m_in) &&
                    ExponentialFamilyProjection.get_projected_to_dims(prj) == size(m_in)
                if matches
                    project_to(prj, generic_logpdf(m_in, z -> loglikelihood(z, i)), m_in)
                else
                    project_to(prj, generic_logpdf(m_in, z -> loglikelihood(z, i) + logpdf(m_in, z)))
                end
            end
            result = FactorizedJoint(projections)
            proposal.distribution = result
            result
        end
    end,
)

end
