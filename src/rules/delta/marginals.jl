@marginalrule DeltaFn{f}(:ins) (q_out::Any, m_ins::NTuple{N, Any}, meta::SamplingApproximation) where {f, N} = begin
    return MvNormalMeanPrecision(zeros(N), diageye(N))
end

@marginalrule DeltaFn{f}(:ins) (m_out::Any, m_ins::NTuple{1, Any}, meta::CVIApproximation) where {f} = begin
    η = naturalParams(m_ins[1])
    logp_nc(z) = (meta.dataset_size / meta.batch_size) * logpdf(m_out, f(z))
    λ = renderCVI(logp_nc, meta.num_iterations, meta.opt, deepcopy(η), m_ins[1])
    meta.q_ins_marginal = FactorProduct((standardDist(λ - η))) 
    return meta.q_ins_marginal
end
