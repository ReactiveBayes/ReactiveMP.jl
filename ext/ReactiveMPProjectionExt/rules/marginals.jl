using TupleTools

import Distributions: Distribution
import BayesBase: AbstractContinuousGenericLogPdf

@marginalrule DeltaFn(:ins) (m_out::Any, m_ins::ManyOf{1, Any}, meta::DeltaMeta{M}) where {M <: CVIProjection} = begin
    g = getnodefn(meta, Val(:out))

    # Create an `AbstractContinuousGenericLogPdf` with an unspecified domain and the transformed `logpdf` function
    F = promote_variate_type(variate_form(typeof(first(m_ins))), BayesBase.AbstractContinuousGenericLogPdf)
    f = convert(F, UnspecifiedDomain(), (z) -> logpdf(m_out, g(z)))

    T = ExponentialFamily.exponential_family_typetag(first(m_ins))
    prj = ProjectedTo(T; parameters = method.prjparams)
    q = project_to(prj, f, first(m_ins))

    return FactorizedJoint((q,))
end

@marginalrule DeltaFn(:ins) (m_out::Any, m_ins::ManyOf{N, Any}, meta::DeltaMeta{M}) where {N, M <: CVIProjection} = begin
    method = ReactiveMP.getmethod(meta)
    rng = method.rng
    pre_samples = zip(map(m_in_k -> ReactiveMP.cvilinearize(rand(rng, m_in_k, method.nsamples)), m_ins)...)

    logp_nc_drop_index = let g = getnodefn(meta, Val(:out)), pre_samples = pre_samples
        (z, i, pre_samples) -> begin
            samples = map(ttuple -> ReactiveMP.TupleTools.insertat(ttuple, i, (z,)), pre_samples)
            t_samples = map(s -> g(s...), samples)
            logpdfs = map(out -> logpdf(m_out, out), t_samples)
            return mean(logpdfs)
        end
    end

    optimize_natural_parameters = let m_ins = m_ins, logp_nc_drop_index = logp_nc_drop_index
        (i, pre_samples) -> begin
            # Create an `AbstractContinuousGenericLogPdf` with an unspecified domain and the transformed `logpdf` function
            df = let i = i, pre_samples = pre_samples, logp_nc_drop_index = logp_nc_drop_index
                (z) -> logp_nc_drop_index(z, i, pre_samples)
            end
            logp = convert(promote_variate_type(variate_form(typeof(first(m_ins))), BayesBase.AbstractContinuousGenericLogPdf), UnspecifiedDomain(), df)

            T = ExponentialFamily.exponential_family_typetag(m_ins[i])
            prj = ProjectedTo(T; parameters = method.prjparams)

            return project_to(prj, logp, m_ins[i])
        end
    end

    return FactorizedJoint(ntuple(i -> optimize_natural_parameters(i, pre_samples), length(m_ins)))
end
