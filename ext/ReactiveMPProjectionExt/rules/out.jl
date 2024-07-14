using ForwardDiff
# cost function
function targetfn(M, p, data)
    ef = convert(ExponentialFamilyDistribution, M, p)
    return -mean(logpdf(ef, data))
end

# gradient function
function grad_targetfn(M, p, data)
    ef = convert(ExponentialFamilyDistribution, M, p)
    fisher = cholinv(ExponentialFamily.fisherinformation(ef))
    return fisher*ForwardDiff.gradient((p) -> targetfn(M, p, data), p)
end

@rule DeltaFn(:out, Marginalisation) (m_out::Union{FactorizedJoint, Uninformative}, q_out::FactorizedJoint, q_ins::FactorizedJoint, meta::DeltaMeta{U}) where {U <: CVIProjection} = begin
    rng                   = Random.default_rng()
    method                = ReactiveMP.getmethod(meta)
    q_ins_components      = components(q_ins)
    dimensions            = map(size, q_ins_components)
    q_ins_sample_friendly = map(q_in -> sampling_optimized(q_in), q_ins_components)
    samples               = map(i -> collect(map(q -> rand(rng, q), q_ins_sample_friendly)), 1:1000)
    node_function         = getnodefn(meta, Val(:out))
    q_out_samples         = mapreduce(sample -> node_function(ReactiveMP.__splitjoin(sample, dimensions)...),hcat, samples)
    q_out_components      = components(q_out)
    Ts                    = map(ExponentialFamily.exponential_family_typetag, q_out_components)
    
    q_out_efs              = map(component -> convert(ExponentialFamilyDistribution, component), q_out_components)
    conditioners           = map(getconditioner, q_out_efs)
    manifolds              = map((T, conditioner, q_out_ef) -> ExponentialFamilyProjection.ExponentialFamilyManifolds.get_natural_manifold(T, size(mean(q_out_ef)), conditioner), Ts, conditioners, q_out_efs)
    natural_parameters_efs = map((m, p) -> ExponentialFamilyProjection.ExponentialFamilyManifolds.partition_point(m,p) ,manifolds, map(getnaturalparameters, q_out_efs))
    ests                   = Vector{ExponentialFamilyDistribution}(undef, length(q_out_efs))
    @inbounds @views for i in eachindex(q_out_efs)
        manifold = manifolds[i]
        naturalparameters =  natural_parameters_efs[i]
        f = let  qsamples = q_out_samples[i, :]
            (M, p) -> begin
                return targetfn(M, p, qsamples)
            end
        end
        g = let qsamples = q_out_samples[i, :] 
            (M, p) -> begin 
                return grad_targetfn(M , p, qsamples)
            end
        end
        ests[i] = convert(ExponentialFamilyDistribution, manifold,
            ExponentialFamilyProjection.Manopt.gradient_descent(manifold, f, g,naturalparameters, stepsize =ExponentialFamilyProjection.Manopt.ConstantStepsize(0.01) )
        )
    end
    
    if typeof(m_out) <: FactorizedJoint
        components_m_out = components(m_out)
        return (x -> logpdf(getindex(ests,k), x) - logpdf(getindex(components_m_out, k), x) for k in eachindex(q_out_efs))
    else
        return (x -> logpdf(getindex(ests,k), x) for k in eachindex(q_out_efs))
    end

end


@rule DeltaFn(:out, Marginalisation) (m_out::Any, q_out::Any, q_ins::FactorizedJoint, meta::DeltaMeta{U}) where {U <: CVIProjection} = begin
    rng                   = Random.default_rng()
    method                = ReactiveMP.getmethod(meta)
    q_ins_components      = components(q_ins)
    dimensions            = map(size, q_ins_components)
    q_ins_sample_friendly = map(q_in -> sampling_optimized(q_in), q_ins_components)
    samples               = map(i -> collect(map(q -> rand(rng, q), q_ins_sample_friendly)), 1:10)
    node_function         = getnodefn(meta, Val(:out))
    q_out_samples         = map(sample -> node_function(ReactiveMP.__splitjoin(sample, dimensions)...), samples)

    T           = ExponentialFamily.exponential_family_typetag(q_out)
    q_out_ef    = convert(ExponentialFamilyDistribution, q_out)
    conditioner = getconditioner(q_out_ef)
    manifold    = ExponentialFamilyProjection.ExponentialFamilyManifolds.get_natural_manifold(T, size(mean(q_out_ef)), conditioner)
    nat_params  = ExponentialFamilyProjection.ExponentialFamilyManifolds.partition_point(manifold,getnaturalparameters(q_out_ef)) 

    f = (M, p) -> targetfn(M, p, q_out_samples)
    g = (M, p) -> grad_targetfn(M , p, q_out_samples)
  
    est = convert(ExponentialFamilyDistribution, manifold,
        ExponentialFamilyProjection.Manopt.gradient_descent(manifold, f, g,nat_params, stepsize =ExponentialFamilyProjection.Manopt.ConstantStepsize(0.01) )
    )
    return x -> logpdf(est, x) - logpdf(m_out, x)
end
