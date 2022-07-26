@marginalrule DeltaFn{f}(:ins) (q_out::Any, m_ins::NTuple{N, Any}, meta::SamplingApproximation) where {f, N} = begin
    return MvNormalMeanPrecision(zeros(N), diageye(N))
end

@marginalrule DeltaFn{f}(:ins) (m_out::Any, m_ins::NTuple{1, Any}, meta::CVIApproximation) where {f} = begin
    η = naturalParams(m_ins[1])
    logp_nc(z) = (meta.dataset_size / meta.batch_size) * logpdf(m_out, f(z))
    λ = renderCVI(logp_nc, meta.num_iterations, meta.opt, meta.rng, deepcopy(η), m_ins[1])
    return FactorProduct((standardDist(λ),))
end

@marginalrule DeltaFn{f}(:ins) (m_out::Any, m_ins::NTuple{N, Any}, meta::CVIApproximation) where {f, N} = begin
    pre_samples = zip([rand(m_ins[i], meta.n_samples) for i in 1:length(m_ins)]...)

    function change_drop_index(ttuple, drop_index, z)
        return (ttuple[1:drop_index-1]..., z, ttuple[drop_index+1:length(ttuple)]...)
    end

    function logp_nc_drop_index(z, i, pre_samples)
        samples = map(ttuple -> change_drop_index(ttuple, i, z), pre_samples)
        t_samples = map(s -> f(s...), samples)
        logpdfs = map(out -> logpdf(m_out, out), t_samples)
        return sum(logpdfs)
    end

    function optimize_natural_parameters(i, pre_samples)
        logp_nc = (z) -> logp_nc_drop_index(z, i, pre_samples)
        return renderCVI(logp_nc, 10, ADAM(), nothing, naturalParams(m_ins[i]), m_ins[i])
    end

    return FactorProduct(Tuple([standardDist(optimize_natural_parameters(i, pre_samples)) for i in 1:length(m_ins)]))
end
