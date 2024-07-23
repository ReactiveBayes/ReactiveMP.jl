using ForwardDiff

@rule DeltaFn(:out, Marginalisation) (m_out::Union{FactorizedJoint, Uninformative}, q_out::FactorizedJoint, q_ins::FactorizedJoint, meta::DeltaMeta{U}) where {U <: CVIProjection} = begin
    node_function         = getnodefn(meta, Val(:out))
    method                = ReactiveMP.getmethod(meta)
    rng                   = getcvirng(method) 
    number_out_samples    = getcvioutsamplesno(method)
    q_ins_components      = components(q_ins)
    dimensions            = map(size, q_ins_components)
    q_ins_sample_friendly = map(q_in -> sampling_optimized(q_in), q_ins_components)
    
    samples               = map(i -> collect(map(q -> rand(rng, q), q_ins_sample_friendly)), 1:number_out_samples)
    q_out_samples         = mapreduce(sample -> node_function(ReactiveMP.__splitjoin(sample, dimensions)...), hcat, samples)
  
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
    rng                   = getcvirng(method) 
    number_out_samples    = getcvioutsamplesno(method)
    q_ins_components      = components(q_ins)
    q_ins_sample_friendly = map(q_in -> sampling_optimized(q_in), q_ins_components)
   
    samples               = map(ReactiveMP.cvilinearize, map(q_in -> rand(rng, q_in, number_out_samples), q_ins_sample_friendly))
    q_out_samples         = map(x -> node_function(x...), zip(samples...))
    
    T           = ExponentialFamily.exponential_family_typetag(q_out)
    q_out_ef    = convert(ExponentialFamilyDistribution, q_out)
    conditioner = getconditioner(q_out_ef)
    manifold    = ExponentialFamilyProjection.ExponentialFamilyManifolds.get_natural_manifold(T, size(mean(q_out_ef)), conditioner)
    nat_params  = ExponentialFamilyProjection.ExponentialFamilyManifolds.partition_point(manifold,getnaturalparameters(q_out_ef)) 

    f = (M, p) -> targetfn(M, p, q_out_samples) 
    g = (M, p) -> grad_targetfn(M , p, q_out_samples)
  
    est = convert(ExponentialFamilyDistribution, manifold,
        ExponentialFamilyProjection.Manopt.gradient_descent(manifold, f, g, nat_params; 
            direction = ExponentialFamilyProjection.BoundedNormUpdateRule(1))
    )
    return DivisionOf(est, m_out)
end


@rule DeltaFn(:out, Marginalisation) (m_ins::ManyOf{1, Any}, meta::DeltaMeta{U}) where {U <: CVIProjection} = begin
    node_function       = getnodefn(meta, Val(:out))
    method              = ReactiveMP.getmethod(meta)
    rng                 = getcvirng(method)
    projection_dims     = getcviprojectiondims(method)
    projection_types    = getcviprojectiontypes(method)
    number_out_samples  = getcvioutsamplesno(method)
    conditioner_out     = getcviprojectionconditioners(method)[:out]
    dim_in              = first(projection_dims[:in])
    T_in                = first(projection_types[:in])
    dim_out             = projection_dims[:out]
    T_out               = projection_types[:out]
    prod_dim_in         = prod(dim_in)
    var_form            = variate_form(T_in)
    
    samples = try
        rand(rng, first(m_ins), number_out_samples)
    catch
        logf                = log_target_adjusted_log_pdf(var_form, first(m_ins), dim_in)
        log_target_density  = LogTargetDensity(prod_dim_in, logf)
        initial_sample      = initialize_cvi_samples(method, rng, first(m_ins), 1, :in)
        vectorized_initial_sample  = vectorize_sample(var_form, initial_sample)
        samples_hmc =  hmc_samples(rng, prod_dim_in, log_target_density, vectorized_initial_sample; no_samples = number_out_samples + 1)
        modify_vectorized_samples_with_variate_type(variate_form(T_in), samples_hmc, dim_in)
    end

    out_samples        = modify_vectorized_samples_with_variate_type(var_form, map(x -> node_function(x), samples), dim_out)

    f = (M, p) -> targetfn(M, p, out_samples)
    g = (M, p) -> grad_targetfn(M, p, out_samples)
    
    out_manifold      = ExponentialFamilyProjection.ExponentialFamilyManifolds.get_natural_manifold(T_out, dim_out, conditioner_out)
    initial_natparams = initialize_cvi_natural_parameters(method, rng, out_manifold,1,:out)
    ef_out = convert(ExponentialFamilyDistribution, out_manifold,
            ExponentialFamilyProjection.Manopt.gradient_descent(out_manifold, f, g, initial_natparams; 
                    direction = ExponentialFamilyProjection.BoundedNormUpdateRule(1))
        )

    dist_out = convert(Distribution, ef_out)
    return dist_out
end

@rule DeltaFn(:out, Marginalisation) (m_ins::ManyOf{N, Any}, meta::DeltaMeta{U}) where {N, U <: CVIProjection} = begin
    node_function       = getnodefn(meta, Val(:out))
    method              = ReactiveMP.getmethod(meta)
    rng                 = getcvirng(method)
    projection_dims     = getcviprojectiondims(method)
    projection_types    = getcviprojectiontypes(method)
    number_out_samples  = getcvioutsamplesno(method)
    conditioner_out     = getcviprojectionconditioners(method)[:out]
    dims_in             = projection_dims[:in]
    Ts_in               = projection_types[:in]
    dim_out             = projection_dims[:out]
    T_out               = projection_types[:out]
    var_form_ins        = variate_form.(Ts_in)
    var_form_out        = variate_form(T_out)
    out_manifold        = ExponentialFamilyProjection.ExponentialFamilyManifolds.get_natural_manifold(T_out, dim_out, conditioner_out)
    prod_dims_in        = map(prod, dims_in)
    sum_dim_in          = sum(prod_dims_in)
    cum_lengths         = mapreduce(d -> d+1, vcat, cumsum(prod_dims_in))
    start_indices       = append!([1], cum_lengths[1:N-1])
    
    # ### Argument to joint logpdf should be vectorized since AdvancedHMC works that way
    joint_logpdf = (x) -> mapreduce((m_in,k,T) -> log_target_adjusted_log_pdf(T, m_in,getindex(dims_in, k))(ReactiveMP.__splitjoinelement(x, getindex(start_indices, k), getindex(dims_in, k))), +, m_ins, 1:N,var_form_ins)
    log_target_density  = LogTargetDensity(sum_dim_in, joint_logpdf)

    initial_sample      = mapreduce((m_in,k) -> initialize_cvi_samples(method, rng, m_in, k, :in),vcat, m_ins, 1:N)
    initial_natparams   = initialize_cvi_natural_parameters(method, rng, out_manifold,1,:out)
        
    samples            = hmc_samples(rng, sum_dim_in, log_target_density, initial_sample; no_samples = number_out_samples + 1)
    out_samples        = modify_vectorized_samples_with_variate_type(var_form_out, map(x -> node_function(ReactiveMP.__splitjoin(x, dims_in)...), samples), dim_out)

    f = (M, p) -> targetfn(M, p, out_samples)
    g = (M, p) -> grad_targetfn(M, p, out_samples)
    
    
    ef_out = convert(ExponentialFamilyDistribution, out_manifold,
            ExponentialFamilyProjection.Manopt.gradient_descent(out_manifold, f, g, initial_natparams; direction = ExponentialFamilyProjection.BoundedNormUpdateRule(1))
        )

    dist_out = convert(Distribution, ef_out)
    return dist_out

end