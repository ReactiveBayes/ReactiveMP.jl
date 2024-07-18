
@rule DeltaFn(:out, Marginalisation) (m_out::Union{FactorizedJoint, Uninformative}, q_out::FactorizedJoint, q_ins::FactorizedJoint, meta::DeltaMeta{U}) where {U <: CVIProjection} = begin
    node_function         = getnodefn(meta, Val(:out))
    method                = ReactiveMP.getmethod(meta)
    rng                   = method.rng
    q_ins_components      = components(q_ins)
    dimensions            = map(size, q_ins_components)
    q_ins_sample_friendly = map(q_in -> sampling_optimized(q_in), q_ins_components)
    
    ## Option 1
    samples               = map(i -> collect(map(q -> rand(rng, q), q_ins_sample_friendly)), 1:method.out_samples_no)
    q_out_samples         = mapreduce(sample -> node_function(ReactiveMP.__splitjoin(sample, dimensions)...),hcat, samples)
    
    ## Option 2
    # samples               = map(ReactiveMP.cvilinearize ,map(q_in -> rand(rng, q_in, method.out_samples_no), q_ins_sample_friendly))
    # q_out_samples         = map(x -> node_function(x...), zip(samples...))
    

    q_out_components      = components(q_out)
    Ts                    = map(ExponentialFamily.exponential_family_typetag, q_out_components)
    
    q_out_efs              = map(component -> convert(ExponentialFamilyDistribution, component), q_out_components)
    conditioners           = map(getconditioner, q_out_efs)
    manifolds              = map((T, conditioner, q_out_ef) -> ExponentialFamilyProjection.ExponentialFamilyManifolds.get_natural_manifold(T, size(mean(q_out_ef)), conditioner), Ts, conditioners, q_out_efs)
    natural_parameters_efs = map((m, p) -> ExponentialFamilyProjection.ExponentialFamilyManifolds.partition_point(m,p) ,manifolds, map(getnaturalparameters, q_out_efs))
    ests                   = Vector{ExponentialFamilyDistribution}(undef, length(q_out_efs))
    
    ## Option 1
    @inbounds @views for i in eachindex(q_out_efs)
        manifold = getindex(manifolds, i)
        naturalparameters = getindex(natural_parameters_efs,i)
        f = let  qsamples = q_out_samples[i, :]
            (M, p) -> begin
                return targetfn(M, p, qsamples)
            end
        end
        g = let  qsamples = q_out_samples[i, :]
            (M, p) -> begin 
                return grad_targetfn(M, p, qsamples)
            end
        end
        ests[i] = convert(ExponentialFamilyDistribution, manifold,
            ExponentialFamilyProjection.Manopt.gradient_descent(manifold, f, g, naturalparameters; direction = ExponentialFamilyProjection.BoundedNormUpdateRule(1))
        )
    end
    
    ## Option 2
    # @inbounds @views for i in eachindex(q_out_efs)
    #     manifold = getindex(manifolds, i)
    #     naturalparameters = getindex(natural_parameters_efs,i)
    #     f = let  qsamples = getindex(q_out_samples, i)
    #         (M, p) -> begin
    #             return targetfn(M, p, qsamples)
    #         end
    #     end
    #     g = let  qsamples = getindex(q_out_samples,i)
    #         (M, p) -> begin 
    #             return grad_targetfn(M, p, qsamples)
    #         end
    #     end
    #     ests[i] = convert(ExponentialFamilyDistribution, manifold,
    #         ExponentialFamilyProjection.Manopt.gradient_descent(manifold, f, g, naturalparameters; direction = ExponentialFamilyProjection.BoundedNormUpdateRule(1))
    #     )
    # end
    
    if typeof(m_out) <: FactorizedJoint
        components_m_out = components(m_out)
        return (DivisionOf(getindex(ests,k), getindex(components_m_out, k)) for k in eachindex(q_out_efs))
    else
        return (DivisionOf(getindex(ests,k), m_out) for k in eachindex(q_out_efs))
    end

   
end


@rule DeltaFn(:out, Marginalisation) (m_out::Any, q_out::Any, q_ins::FactorizedJoint, meta::DeltaMeta{U}) where {U <: CVIProjection} = begin
    node_function         = getnodefn(meta, Val(:out))
    method                = ReactiveMP.getmethod(meta)
    rng                   = method.rng
    q_ins_components      = components(q_ins)
    dimensions            = map(size, mean.(q_ins_components))
    q_ins_sample_friendly = map(q_in -> sampling_optimized(q_in), q_ins_components)
   
    samples               = map(ReactiveMP.cvilinearize ,map(q_in -> rand(rng, q_in, method.out_samples_no), q_ins_sample_friendly))
    @show samples
    q_out_samples         = map(x -> node_function(x...), zip(samples...))
    @show q_out_samples
    
    T           = ExponentialFamily.exponential_family_typetag(q_out)
    q_out_ef    = convert(ExponentialFamilyDistribution, q_out)
    conditioner = getconditioner(q_out_ef)
    manifold    = ExponentialFamilyProjection.ExponentialFamilyManifolds.get_natural_manifold(T, size(mean(q_out_ef)), conditioner)
    nat_params  = ExponentialFamilyProjection.ExponentialFamilyManifolds.partition_point(manifold,getnaturalparameters(q_out_ef)) 

    f = (M, p) -> targetfn(M, p, q_out_samples) 
    g = (M, p) -> grad_targetfn(M , p, q_out_samples)
  
    est = convert(ExponentialFamilyDistribution, manifold,
        ExponentialFamilyProjection.Manopt.gradient_descent(manifold, f, g, nat_params; direction = ExponentialFamilyProjection.BoundedNormUpdateRule(1))
    )
    @show convert(Distribution, est)
    return DivisionOf(est, m_out)
end
