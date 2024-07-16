using TupleTools

import Distributions: Distribution
import BayesBase: AbstractContinuousGenericLogPdf



@marginalrule DeltaFn(:ins) (m_out::Any, m_ins::ManyOf{1, Any}, meta::DeltaMeta{M}) where {M <: CVIProjection} = begin
    g = getnodefn(meta, Val(:out))
    method = ReactiveMP.getmethod(meta)
    var_form = variate_form(typeof(first(m_ins)))
    F = promote_variate_type(var_form, BayesBase.AbstractContinuousGenericLogPdf)
    f = convert(F, UnspecifiedDomain(), (z) -> logpdf(m_out, g(z))+ logpdf(first(m_ins),z))

    
    T = ExponentialFamily.exponential_family_typetag(first(m_ins))
    projection_parameters = method.projection_parameters
    if var_form <: Univariate
        prj = ProjectedTo(T ; parameters = projection_parameters)
    elseif var_form <: Multivariate
        prj = ProjectedTo(T, length(mean(first(m_ins))); parameters = projection_parameters)
    elseif var_form <: Matrixvariate
        prj = ProjectedTo(T, size(mean(first(m_ins)))...; parameters = projection_parameters)
    end
    q = project_to(prj, f)

    return FactorizedJoint((q,))
end

@marginalrule DeltaFn(:ins) (m_out::Any, m_ins::ManyOf{N, Any}, meta::DeltaMeta{M}) where {N, M <: CVIProjection} = begin
    method = ReactiveMP.getmethod(meta)
    rng = method.rng
    projection_parameters = method.projection_parameters
    pre_samples = zip(map(m_in_k -> ReactiveMP.cvilinearize(rand(rng, m_in_k, method.marginal_samples_no)), m_ins)...)

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
            df = let i = i, pre_samples = pre_samples, logp_nc_drop_index = logp_nc_drop_index
                (z) -> logp_nc_drop_index(z, i, pre_samples) 
            end
            T = ExponentialFamily.exponential_family_typetag(m_ins[i])
            var_form = variate_form(typeof(m_ins[i]))
            logp = convert(promote_variate_type(var_form, BayesBase.AbstractContinuousGenericLogPdf), UnspecifiedDomain(), df)
            if var_form <: Univariate
                prj = ProjectedTo(T ; parameters = projection_parameters)
            elseif var_form <: Multivariate
                prj = ProjectedTo(T, length(mean(m_ins[i])); parameters = projection_parameters)
            elseif var_form <: Matrixvariate
                prj = ProjectedTo(T, size(mean(m_ins[i]))...; parameters = projection_parameters)
            end
          
            projection_result =  project_to(prj, logp, m_ins[i])
            return projection_result
        end
    end
    result = FactorizedJoint(ntuple(i -> optimize_natural_parameters(i, pre_samples), length(m_ins)))
    return result
end
