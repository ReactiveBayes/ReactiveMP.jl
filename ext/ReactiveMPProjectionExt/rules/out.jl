using ForwardDiff

# cost function
function targetfn(M, p, data)
    ef = convert(ExponentialFamilyDistribution, M, p)
    return -sum((d) -> logpdf(ef, d), data)
end

# # gradient function
## I think this is wrong. This is not a gradient on the manifolds. It is just Euclidean gradient.
function grad_targetfn(M, p, data)
    ef = convert(ExponentialFamilyDistribution, M, p)
    ifisher = cholinv(Hermitian(fisherinformation(ef)))
    X = ExponentialFamilyProjection.ExponentialFamilyManifolds.partition_point(M, ifisher * ForwardDiff.gradient((p) -> targetfn(M, p, data), p))
    X = ExponentialFamilyProjection.ExponentialFamilyManifolds.ManifoldsBase.project(M, p, X)
    return X
end

@rule DeltaFn(:out, Marginalisation) (m_out::Any, q_out::Any, q_ins::FactorizedJoint, meta::DeltaMeta{U}) where {U <: CVIProjection} = begin
    node_function         = getnodefn(meta, Val(:out))
    method                = ReactiveMP.getmethod(meta)
    rng                   = method.rng
    q_ins_components      = components(q_ins)
    dimensions            = map(size, q_ins_components)
    q_ins_sample_friendly = map(q_in -> sampling_optimized(q_in), q_ins_components)
    ## Option 1
    # samples               = map(i -> collect(map(q -> rand(rng, q), q_ins_sample_friendly)), 1:method.out_samples_no)
    # q_out_samples         = map(sample -> node_function(ReactiveMP.__splitjoin(sample, dimensions)...), samples)

    ## Option 2
    samples       = map(ReactiveMP.cvilinearize, map(q_in -> rand(rng, q_in, method.outsamples), q_ins_sample_friendly))
    q_out_samples = map(x -> node_function(x...), zip(samples...))

    ## Option 3
    # T = ExponentialFamily.exponential_family_typetag(q_out)
    # s = sampling_optimized(q_out)
    # d = fit_mle(typeof(s), q_out_samples)
    # m = DivisionOf(d, m_out)
    # r = project_to(ProjectedTo(T, size(q_out)...; parameters = method.prjparams), (x) -> logpdf(m, x))
    # return r

    T           = ExponentialFamily.exponential_family_typetag(q_out)
    q_out_ef    = convert(ExponentialFamilyDistribution, q_out)
    conditioner = getconditioner(q_out_ef)
    manifold    = ExponentialFamilyProjection.ExponentialFamilyManifolds.get_natural_manifold(T, size(mean(q_out_ef)), conditioner)
    nat_params  = ExponentialFamilyProjection.ExponentialFamilyManifolds.partition_point(manifold, getnaturalparameters(q_out_ef))

    f = (M, p) -> targetfn(M, p, q_out_samples)
    g = (M, p) -> grad_targetfn(M, p, q_out_samples)

    est = convert(
        ExponentialFamilyDistribution,
        manifold,
        ExponentialFamilyProjection.Manopt.gradient_descent(
            manifold, f, g, nat_params; 
            stepsize = ExponentialFamilyProjection.Manopt.ConstantStepsize(0.1), 
            direction = ExponentialFamilyProjection.BoundedNormUpdateRule(1)
        )
    )
    # return x -> logpdf(est, x) - logpdf(m_out, x)
    return DivisionOf(est, m_out)
end