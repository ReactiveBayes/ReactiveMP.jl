# First method: when there's no projection form
function create_project_to(method::CVIProjection{R, S, P, Nothing}, q_out::Any, q_out_samples::Any) where {R, S, P}
    T = ExponentialFamily.exponential_family_typetag(q_out)
    q_out_ef    = convert(ExponentialFamilyDistribution, q_out)
    conditioner = getconditioner(q_out_ef)
    return ProjectedTo(T, size(first(q_out_samples))...; conditioner = conditioner, parameters = something(method.prjparams, ExponentialFamilyProjection.DefaultProjectionParameters()))
end

function create_project_to(method::CVIProjection{R, S, P, F}, ::Any, ::Any) where {R, S, P, F <: ProjectionForm}
    form = method.target_out_form
    return ProjectedTo(form.typeform, form.dims...; conditioner = form.conditioner, parameters = something(method.prjparams, ExponentialFamilyProjection.DefaultProjectionParameters()))
end

@rule DeltaFn(:out, Marginalisation) (m_out::Any, q_out::Any, q_ins::FactorizedJoint, meta::DeltaMeta{U}) where {U <: CVIProjection} = begin
    node_function         = getnodefn(meta, Val(:out))
    method               = ReactiveMP.getmethod(meta)
    rng                  = method.rng
    q_ins_components     = components(q_ins)
    q_ins_sample_friendly = map(q_in -> sampling_optimized(q_in), q_ins_components)
  
    samples       = map(ReactiveMP.cvilinearize, map(q_in -> rand(rng, q_in, method.outsamples), q_ins_sample_friendly))
    q_out_samples = map(x -> node_function(x...), zip(samples...))

    prj = create_project_to(method, q_out, q_out_samples)
    est = project_to(prj, q_out_samples)    
    return DivisionOf(est, m_out)
end