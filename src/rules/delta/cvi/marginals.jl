using TupleTools
import Distributions: Distribution

@marginalrule DeltaFn{f}(:ins) (m_out::Any, m_ins::ManyOf{1, Any}, meta::CVIApproximation) where {f} = begin
    η = naturalparams(m_ins[1])
    logp_nc = (z) -> logpdf(m_out, f(z))
    λ = renderCVI(logp_nc, meta.num_iterations, meta.opt, meta.rng, deepcopy(η), m_ins[1])
    return FactorProduct((convert(Distribution, λ),))
end

@marginalrule DeltaFn{f}(:ins) (m_out::Any, m_ins::ManyOf{N, Any}, meta::CVIApproximation) where {f, N} = begin
    rng = something(meta.rng, Random.GLOBAL_RNG)
    pre_samples = zip(map(m_in_k -> cvilinearize(rand(rng, m_in_k, meta.n_samples)), m_ins)...)

    logp_nc_drop_index =
        (z, i, pre_samples) -> begin
            samples = map(ttuple -> TupleTools.insertat(ttuple, i, (z,)), pre_samples)
            t_samples = map(s -> f(s...), samples)
            logpdfs = map(out -> logpdf(m_out, out), t_samples)
            return mean(logpdfs)
        end

    optimize_natural_parameters =
        (i, pre_samples) -> begin
            logp_nc = (z) -> logp_nc_drop_index(z, i, pre_samples)
            return renderCVI(logp_nc, meta.num_iterations, meta.opt, meta.rng, naturalparams(m_ins[i]), m_ins[i])
        end

    return FactorProduct(ntuple(i -> convert(Distribution, optimize_natural_parameters(i, pre_samples)), length(m_ins)))
end
