using TupleTools, AdvancedHMC, LogDensityProblems

import Distributions: Distribution
import BayesBase: AbstractContinuousGenericLogPdf



@marginalrule DeltaFn(:ins) (m_out::Any, m_ins::ManyOf{1, Any}, meta::DeltaMeta{M}) where {M <: CVIProjection} = begin
    method = ReactiveMP.getmethod(meta)
    g = getnodefn(meta, Val(:out))

    m_in = first(m_ins)
    # Create an `AbstractContinuousGenericLogPdf` with an unspecified domain and the transformed `logpdf` function
    F = promote_variate_type(variate_form(typeof(m_in)), BayesBase.AbstractContinuousGenericLogPdf)
    f = convert(F, UnspecifiedDomain(), (z) -> logpdf(m_out, g(z)))

    T = ExponentialFamily.exponential_family_typetag(m_in)
    prj = ProjectedTo(T, size(m_in)...; parameters = method.projection_parameters)
    q = project_to(prj, f, first(m_ins))

    return FactorizedJoint((q,))
end


@marginalrule DeltaFn(:ins) (m_out::Any, m_ins::ManyOf{N, Any}, meta::DeltaMeta{M}) where {N, M <: CVIProjection} = begin
    method                = ReactiveMP.getmethod(meta)
    rng                   = method.rng
    node_function         = getnodefn(meta, Val(:out))
    means_m_ins   = map(mean, m_ins)
    dims          = map(size, means_m_ins)
    lengths       = mapreduce(length, vcat, means_m_ins)
    d             = sum(lengths)
    cum_lengths   = map(d -> d+1, cumsum(lengths))
    start_indices = append!([1], cum_lengths[1:N-1])

    
    ### for HMC
    joint_logpdf = (x) -> logpdf(m_out, node_function(ReactiveMP.__splitjoin(x, dims)...)) + mapreduce((m_in,k) -> logpdf(m_in, ReactiveMP.__splitjoinelement(x, k, getindex(dims, k))) , +, m_ins, 1:N)
    
    log_target_density = LogTargetDensity(d, joint_logpdf)
    
    initial_x = mapreduce(m_in -> rand(rng, m_in), vcat, m_ins)
    
    n_adapts = 1_000
    
    metric = AdvancedHMC.DiagEuclideanMetric(d) ### We should use fisher metric here
    hamiltonian = AdvancedHMC.Hamiltonian(metric, log_target_density, ForwardDiff)
    initial_ϵ = AdvancedHMC.find_good_stepsize(hamiltonian, initial_x)
    integrator = AdvancedHMC.Leapfrog(initial_ϵ)
    

    kernel = AdvancedHMC.HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
    adaptor = AdvancedHMC.StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
    samples, _ = AdvancedHMC.sample(rng, hamiltonian, kernel, initial_x, method.marginal_samples_no+1, adaptor, n_adapts; verbose = false, progress=false)
    samples_matrix = map(k -> map((sample) -> ReactiveMP.__splitjoinelement(sample, getindex(start_indices,k), getindex(dims, k)), samples[2:end]), 1:N)
    
    m_ins_efs              = map(component -> convert(ExponentialFamilyDistribution, component), m_ins)
    Ts                     = map(ExponentialFamily.exponential_family_typetag, m_ins_efs)
    conditioners           = map(getconditioner, m_ins_efs)
    manifolds              = map((T, conditioner,m_in_ef) -> ExponentialFamilyProjection.ExponentialFamilyManifolds.get_natural_manifold(T, size(mean(m_in_ef)), conditioner), Ts, conditioners, m_ins_efs)
    natural_parameters_efs = map((m, p) -> ExponentialFamilyProjection.ExponentialFamilyManifolds.partition_point(m,p) ,manifolds, map(getnaturalparameters, m_ins_efs))

    function projection_to_ef(i)
        manifold = getindex(manifolds, i)
        naturalparameters = getindex(natural_parameters_efs,i)
        f = let @views sample = samples_matrix[i]
            (M, p) -> begin
                return targetfn(M, p, sample)
            end
        end
        g = let @views sample = samples_matrix[i]
            (M, p) -> begin 
                return grad_targetfn(M, p, sample)
            end
        end
        return convert(ExponentialFamilyDistribution, manifold,
            ExponentialFamilyProjection.Manopt.gradient_descent(manifold, f, g, naturalparameters; direction = ExponentialFamilyProjection.BoundedNormUpdateRule(1))
        )
     
    end
    result = FactorizedJoint(map(d -> convert(Distribution, d), ntuple(i -> projection_to_ef(i), N)))
    return result

end
