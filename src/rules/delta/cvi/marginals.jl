using TupleTools
import Distributions: Distribution

@marginalrule DeltaFn(:ins) (m_out::Any, m_ins::ManyOf{1, Any}, meta::DeltaMeta{M}) where {M <: CVI} = begin
    g = getnodefn(Val(:out))
    q = convert(Distribution, prod(getmethod(meta), (z) -> logpdf(m_out, g(z)), first(m_ins)))
    return FactorizedJoint((q,))
end

@marginalrule DeltaFn(:ins) (m_out::Any, m_ins::ManyOf{N, Any}, meta::DeltaMeta{M}) where {N, M <: CVI} = begin
    method = getmethod(meta)
    rng = something(method.rng, Random.GLOBAL_RNG)
    pre_samples = zip(map(m_in_k -> cvilinearize(rand(rng, m_in_k, method.n_samples)), m_ins)...)

    logp_nc_drop_index = let g = getnodefn(Val(:out))
        (z, i, pre_samples) -> begin
            samples = map(ttuple -> TupleTools.insertat(ttuple, i, (z,)), pre_samples)
            t_samples = map(s -> g(s...), samples)
            logpdfs = map(out -> logpdf(m_out, out), t_samples)
            return mean(logpdfs)
        end
    end

    optimize_natural_parameters = (i, pre_samples) -> prod(method, (z) -> logp_nc_drop_index(z, i, pre_samples), m_ins[i])

    return FactorizedJoint(ntuple(i -> convert(Distribution, optimize_natural_parameters(i, pre_samples)), length(m_ins)))
end
