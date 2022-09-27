using TupleTools
import Distributions: Distribution

@marginalrule DeltaFn{f}(:ins) (m_out::Any, m_ins::ManyOf{1, Any}, meta::CVIApproximation) where {f} = begin
    η = naturalparams(m_ins[1])
    logp_nc(z) = logpdf(m_out, f(z))
    λ = renderCVI(logp_nc, meta.num_iterations, meta.opt, meta.rng, deepcopy(η), m_ins[1])
    return FactorProduct((convert(Distribution, λ),))
end

@marginalrule DeltaFn{f}(:ins) (m_out::Any, m_ins::ManyOf{N, Any}, meta::CVIApproximation) where {f, N} = begin
    pre_samples = zip([rand(m_ins[i], meta.n_samples) for i in 1:length(m_ins)]...)

    logp_nc_drop_index =
        (z, i, pre_samples) -> begin
            samples = map(ttuple -> TupleTools.insertat(ttuple, i, (z,)), pre_samples)
            t_samples = map(s -> f(s...), samples)
            logpdfs = map(out -> logpdf(m_out, out), t_samples)
            return sum(logpdfs)
        end

    optimize_natural_parameters =
        (i, pre_samples) -> begin
            logp_nc = (z) -> logp_nc_drop_index(z, i, pre_samples)
            return renderCVI(logp_nc, meta.num_iterations, meta.opt, meta.rng, naturalparams(m_ins[i]), m_ins[i])
        end

    return FactorProduct(ntuple(i -> optimize_natural_parameters(i, pre_samples), length(m_ins)))
end
