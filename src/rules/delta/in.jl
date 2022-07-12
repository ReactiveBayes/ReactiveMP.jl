using Random

@rule DeltaFn{f}((:in, k), Marginalisation) (q_ins::Any, m_in::Any, meta::SamplingApproximation) where {f} = begin
    return ContinuousUnivariateLogPdf((x) -> logpdf(m_in, f(x)))
end

@rule DeltaFn{f}((:in, k), Marginalisation) (
    q_ins::Any,
    m_in::Any,
    meta::LinearApproximationKnownInverse
) where {f} = begin
    return NormalMeanVariance(0, 1)
end

@rule DeltaFn{f}((:in, k), Marginalisation) (q_ins::FactorProduct, m_in::Any, meta::CVIApproximation) where {f} = begin
    return q_ins[k]
end

# @rule function DeltaFn{f}((:in, k), Marginalisation)
#     (msg_out, msg_in, meta::CVIApproximation)
#     η = naturalParams(msg_in)
#     λ_init = deepcopy(η)

#     logp_nc(z) = (meta.dataset_size / meta.batch_size) * logPdf(msg_out, f(z))
#     λ = renderCVI(logp_nc, meta.num_iterations, meta.opt, λ_init, msg_in)

#     λ_message = λ - η

#     return standardDist(λ_message)
# end
