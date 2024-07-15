
# This function performs MLE estimation for `q_out` given a set of samples
function rule_q_out_cvi_projectio(rng, prj, samples, starting_point)
    f = let samples = samples
        (M, p) -> begin
            ef = convert(ExponentialFamilyDistribution, M, p)
            return -sum((d) -> logpdf(ef, d), samples)
        end
    end

    g = let f = f
        (M, p) -> begin
            X = ReactiveMP.ForwardDiff.gradient((p) -> f(M, p), p)
            X = ExponentialFamilyProjection.ExponentialFamilyManifolds.partition_point(M, X)
            N = norm(M, p, X)
            if N > one(N)
                X = X ./ N
            end
            return X
        end
    end

    M = ExponentialFamilyProjection.get_projected_to_manifold(prj)
    p = rand(rng, M)
    p .= starting_point
    q = ExponentialFamilyProjection.Manopt.gradient_descent(M, f, g, p; stepsize = ExponentialFamilyProjection.Manopt.ConstantStepsize(0.1), debug = missing)

    return convert(ExponentialFamilyDistribution, M, q)
end

@rule DeltaFn(:out, Marginalisation) (m_out::Any, q_out::Any, q_ins::FactorizedJoint{P}, meta::DeltaMeta{U}) where {P <: NTuple{1}, U <: CVIProjection} = begin
    method            = ReactiveMP.getmethod(meta)
    rng               = method.rng
    q_sample_friendly = sampling_optimized(q_ins[1])
    samples           = map(x -> rand(rng, q_sample_friendly), 1:(method.nsamples))
    q_out_samples     = map(getnodefn(meta, Val(:out)), samples)

    T = ExponentialFamily.exponential_family_typetag(q_out)
    q_out_ef = convert(ExponentialFamilyDistribution, q_out)
    c = getconditioner(q_out_ef)
    prj = ProjectedTo(T, size(mean(q_out_ef))...; conditioner = c)

    # r = sampling_optimized(q_out)
    # est = fit_mle(typeof(r), q_out_samples)

    q_out = rule_q_out_cvi_projectio(rng, prj, q_out_samples, getnaturalparameters(q_out_ef))

    return DivisionOf(q_out, m_out)
end

@rule DeltaFn(:out, Marginalisation) (m_out::Any, q_out::Any, q_ins::FactorizedJoint, meta::DeltaMeta{M}) where {M <: CVIProjection} = begin
    method = ReactiveMP.getmethod(meta)
    rng = method.rng
    q_ins_sample_friendly = map(marginal -> sampling_optimized(marginal), components(q_ins))
    q_ins_samples = map(marginal -> rand(rng, marginal, method.nsamples), q_ins_sample_friendly)
    samples_linear = map(ReactiveMP.cvilinearize, q_ins_samples)
    g = getnodefn(meta, Val(:out))
    q_out_samples = map(x -> g(x...), zip(samples_linear...))

    T = ExponentialFamily.exponential_family_typetag(q_out)
    q_out_ef = convert(ExponentialFamilyDistribution, q_out)
    c = getconditioner(q_out_ef)
    prj = ProjectedTo(T, size(mean(q_out_ef))...; conditioner = c)

    q_out = rule_q_out_cvi_projectio(rng, prj, q_out_samples, getnaturalparameters(q_out_ef))
    return DivisionOf(q_out, m_out)
end