using TupleTools, AdvancedHMC, LogDensityProblems

import Distributions: Distribution
import BayesBase: AbstractContinuousGenericLogPdf



@marginalrule DeltaFn(:ins) (m_out::Any, m_ins::ManyOf{1, Any}, meta::DeltaMeta{M}) where {M <: CVIProjection} = begin
    method = ReactiveMP.getmethod(meta)
    g      = getnodefn(meta, Val(:out))
    m_in   = first(m_ins)
    T  = try 
            ExponentialFamily.exponential_family_typetag(m_in)
        catch e
            first(getcviprojectiontypes(method)[:in])
        end
    prj = ProjectedTo(T, size(m_in)...; parameters = getcviprojectionparameters(method))
    q = project_to(prj, (z) -> logpdf(m_out, g(z)), m_in)
    # q = project_to(prj, (z) -> logpdf(m_out, g(z)) + logpdf(m_in,z)) ## this is still valid if m_in is not in ef

    return FactorizedJoint((q,))
end


@marginalrule DeltaFn(:ins) (m_out::Any, m_ins::ManyOf{N, Any}, meta::DeltaMeta{M}) where {N, M <: CVIProjection} = begin
    node_function       = getnodefn(meta, Val(:out))
    method              = ReactiveMP.getmethod(meta)
    rng                 = getcvirng(method)
    number_marginal_samples  = getcvimarginalsamplesno(method)

    m_ins_efs = try
        map(component -> convert(ExponentialFamilyDistribution, component), m_ins)
    catch
        nothing
    end

    if !isnothing(m_ins_efs)
        var_form_ins           = map(d -> variate_form(typeof(d)), m_ins)
        dims_in                = map(size, m_ins)
        prod_dims_in           = map(prod, dims_in)
        sum_dim_in             = sum(prod_dims_in)
        cum_lengths            = mapreduce(d -> d+1, vcat, cumsum(prod_dims_in))
        start_indices          = append!([1], cum_lengths[1:N-1])
        Ts                     = map(ExponentialFamily.exponential_family_typetag, m_ins_efs)
        conditioners           = map(getconditioner, m_ins_efs)
        manifolds              = map((T, conditioner,m_in_ef) -> ExponentialFamilyProjection.ExponentialFamilyManifolds.get_natural_manifold(T, size(mean(m_in_ef)), conditioner), Ts, conditioners, m_ins_efs)
        natural_parameters_efs = map((m, p) -> ExponentialFamilyProjection.ExponentialFamilyManifolds.partition_point(m,p) ,manifolds, map(getnaturalparameters, m_ins_efs))
        initial_sample         = mapreduce((m_in,k) -> initialize_cvi_samples(method, rng, m_in, k, :in),vcat, m_ins, 1:N)
    else
        var_form_ins        = map(variate_form, getcviprojectiontypes(method)[:in])
        dims_in             = getcviprojectiondims(method)[:in]
        prod_dims_in        = map(prod, dims_in)
        sum_dim_in          = sum(prod_dims_in)
        cum_lengths         = mapreduce(d -> d+1, vcat, cumsum(prod_dims_in))
        start_indices       = append!([1], cum_lengths[1:N-1])
        conditioners        = getcviprojectionconditioners(method)[:in]
        Ts                  = getcviprojectiontypes(method)[:in]
        manifolds = if !isnothing(conditioners)
                map((T, conditioner, dim) -> ExponentialFamilyProjection.ExponentialFamilyManifolds.get_natural_manifold(T, dim, conditioner), Ts, conditioners, dims_in)
            else
                map((T, dim) -> ExponentialFamilyProjection.ExponentialFamilyManifolds.get_natural_manifold(T, dim), Ts, dims_in)
            end
        natural_parameters_efs = map((manifold,k) -> initialize_cvi_natural_parameters(method, rng, manifold, k, :in), manifolds, 1:N)
        initial_sample         = rand(rng, sum_dim_in)
    end

    joint_logpdf = (x) -> logpdf(m_out, node_function(ReactiveMP.__splitjoin(x, dims_in)...)) + mapreduce((m_in,k,T) -> log_target_adjusted_log_pdf(T, m_in, getindex(dims_in, k))(ReactiveMP.__splitjoinelement(x, getindex(start_indices, k), getindex(dims_in, k))), +, m_ins, 1:N, var_form_ins)
    log_target_density  = LogTargetDensity(sum_dim_in, joint_logpdf)

    samples            = hmc_samples(rng, sum_dim_in, log_target_density, initial_sample; no_samples = number_marginal_samples + 1)
    samples_matrix     = map(k -> map((sample) -> ReactiveMP.__splitjoinelement(sample, getindex(start_indices,k), getindex(dims_in, k)), samples), 1:N)
    
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

