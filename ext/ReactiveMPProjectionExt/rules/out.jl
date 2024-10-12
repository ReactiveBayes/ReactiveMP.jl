@rule DeltaFn(:out, Marginalisation) (m_out::Any, q_out::Any, q_ins::FactorizedJoint, meta::DeltaMeta{U}) where {U <: CVIProjection} = begin
    node_function         = getnodefn(meta, Val(:out))
    method                = ReactiveMP.getmethod(meta)
    rng                   = method.rng
    q_ins_components      = components(q_ins)
    q_ins_sample_friendly = map(q_in -> sampling_optimized(q_in), q_ins_components)
  
    samples       = map(ReactiveMP.cvilinearize, map(q_in -> rand(rng, q_in, method.outsamples), q_ins_sample_friendly))
    q_out_samples = map(x -> node_function(x...), zip(samples...))

    T           = ExponentialFamily.exponential_family_typetag(q_out)
    q_out_ef    = convert(ExponentialFamilyDistribution, q_out)
    conditioner = getconditioner(q_out_ef)
    
    prj = ProjectedTo(T, size(first(q_out_samples))...; conditioner = conditioner, parameters = something(method.prjparams, ExponentialFamilyProjection.DefaultProjectionParameters()))
    est  = project_to(prj, q_out_samples)
    
    return DivisionOf(est, m_out)
end