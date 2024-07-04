@rule DeltaFn(:out, Marginalisation) (m_out::Any, q_out::Any, q_ins::FactorizedJoint{P}, meta::DeltaMeta{U}) where {P <: NTuple{1}, U <: CVIProjection} = begin
    
    rng = StableRNG(42)
    method            = ReactiveMP.getmethod(meta)
    q_sample_friendly = sampling_optimized(q_ins[1])
    samples           = map(x -> rand(rng, q_sample_friendly), 1:10)
    q_out_samples     = map(getnodefn(meta, Val(:out)), samples)

    #label-1
    
    T = ExponentialFamily.exponential_family_typetag(q_out)
    q_out_ef = convert(ExponentialFamilyDistribution, q_out)
    c = getconditioner(q_out_ef)
    prj = ProjectedTo(T, size(mean(q_out_ef))...; conditioner = c)

    f = let m_out = m_out
        (M, p) -> begin
            ef = convert(ExponentialFamilyDistribution, M, p)
            return -sum((d) -> logpdf(ef, d) - logpdf(m_out, d), q_out_samples)
        end
    end

    g = let f = f 
        (M, p) -> begin
            ef = convert(ExponentialFamilyDistribution, M, p)
            X = ReactiveMP.ForwardDiff.gradient((p) -> f(M, p), p)
            if !LinearAlgebra.isposdef(ExponentialFamily.fisherinformation(ef))
                @show ef, convert(Distribution, ef), q_out, p
            end
            inv_fisher = cholinv(ExponentialFamily.fisherinformation(ef))
            X = inv_fisher * X
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
    p .= getnaturalparameters(q_out_ef)
    q = ExponentialFamilyProjection.Manopt.gradient_descent(M, f, g, p; 
        stepsize = ExponentialFamilyProjection.Manopt.ConstantStepsize(0.1),
        debug = missing
    )

    qef = convert(ExponentialFamilyDistribution, M, q)
    
    η = getnaturalparameters(qef)
    
    result = ExponentialFamilyDistribution(T, η, c, nothing)
    return convert(Distribution, result)
end

@rule DeltaFn(:out, Marginalisation) (q_ins::FactorizedJoint, meta::DeltaMeta{M}) where {M <: CVIProjection} = begin
    error("resive this")
    method = ReactiveMP.getmethod(meta)
    q_ins_sample_friendly = map(marginal -> sampling_optimized(marginal), components(q_ins))
    rng = something(StableRNG(42), Random.default_rng())
    q_ins_samples = map(marginal -> rand(rng, marginal, 10), q_ins_sample_friendly)
    samples_linear = map(ReactiveMP.cvilinearize, q_ins_samples)
    g = getnodefn(meta, Val(:out))
    samples = map(x -> g(x...), zip(samples_linear...))
    return TerminalProdArgument(SampleList(samples))
end