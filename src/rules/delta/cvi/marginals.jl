using TupleTools
import Distributions: Distribution

@marginalrule DeltaFn{f}(:ins) (m_out::Any, m_ins::ManyOf{1, Any}, meta::CVIApproximation) where {f} = begin
    q = convert(Distribution, render_cvi(meta, (z) -> logpdf(m_out, f(z)), first(m_ins)))
    return FactorizedJoint((q,))
end

@marginalrule DeltaFn{f}(:ins) (m_out::Any, m_ins::ManyOf{N, Any}, meta::CVIApproximation) where {f, N} = begin
    rng = something(meta.rng, Random.GLOBAL_RNG)
    pre_samples = zip(map(m_in_k -> cvilinearize(rand(rng, m_in_k, meta.n_samples)), m_ins)...)

    logp_nc_drop_index = (z, i, pre_samples) -> begin
        samples = map(ttuple -> TupleTools.insertat(ttuple, i, (z,)), pre_samples)
        t_samples = map(s -> f(s...), samples)
        logpdfs = map(out -> logpdf(m_out, out), t_samples)
        return mean(logpdfs)
    end

    optimize_natural_parameters = (i, pre_samples) -> render_cvi(meta, (z) -> logp_nc_drop_index(z, i, pre_samples), m_ins[i])

    return FactorizedJoint(ntuple(i -> convert(Distribution, optimize_natural_parameters(i, pre_samples)), length(m_ins)))
end
