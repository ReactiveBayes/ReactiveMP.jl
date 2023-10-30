using TupleTools

import Distributions: Distribution
import BayesBase: AbstractContinuousGenericLogPdf


@marginalrule DeltaFn(:ins) (m_out::Any, m_ins::ManyOf{1, Any}, meta::DeltaMeta{M}) where {M <: CVI} = begin
    g = getnodefn(Val(:out))

    # Create an `AbstractContinuousGenericLogPdf` with an unspecified domain and the transformed `logpdf` function
    F = promote_variate_type(variate_form(typeof(first(m_ins))), AbstractContinuousGenericLogPdf)
    f = convert(F, UnspecifiedDomain(), (z) -> logpdf(m_out, g(z)))
    q = prod(getmethod(meta), f, first(m_ins))

    return FactorizedJoint((q,))
end

@marginalrule DeltaFn(:ins) (m_out::Any, m_ins::ManyOf{N, Any}, meta::DeltaMeta{M}) where {N, M <: CVI} = begin
    method = getmethod(meta)
    rng = something(method.rng, Random.GLOBAL_RNG)
    pre_samples = zip(map(m_in_k -> cvilinearize(rand(rng, m_in_k, method.n_samples)), m_ins)...)

    logp_nc_drop_index = let g = getnodefn(Val(:out)), pre_samples = pre_samples
        (z, i, pre_samples) -> begin
            samples = map(ttuple -> TupleTools.insertat(ttuple, i, (z,)), pre_samples)
            t_samples = map(s -> g(s...), samples)
            logpdfs = map(out -> logpdf(m_out, out), t_samples)
            return mean(logpdfs)
        end
    end

    optimize_natural_parameters = let method = method, m_ins = m_ins, logp_nc_drop_index = logp_nc_drop_index
        (i, pre_samples) -> begin
            # Create an `AbstractContinuousGenericLogPdf` with an unspecified domain and the transformed `logpdf` function
            df = let i = i, pre_samples = pre_samples, logp_nc_drop_index = logp_nc_drop_index
                (z) -> logp_nc_drop_index(z, i, pre_samples)
            end
            logp = convert(promote_variate_type(variate_form(typeof(first(m_ins))), AbstractContinuousGenericLogPdf), UnspecifiedDomain(), df)
            return prod(method, logp, m_ins[i])
        end
    end

    return FactorizedJoint(ntuple(i -> optimize_natural_parameters(i, pre_samples), length(m_ins)))
end
