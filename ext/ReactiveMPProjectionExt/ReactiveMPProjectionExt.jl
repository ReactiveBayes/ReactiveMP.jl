module ReactiveMPProjectionExt

using ReactiveMP, ExponentialFamily, AdvancedHMC, LogDensityProblems, Distributions, ExponentialFamilyProjection, BayesBase, Random, LinearAlgebra, FastCholesky
using ForwardDiff

export DivisionOf, CVIProjection, CVIProjectionEssentials, CVIProjectionOptional, LogTargetDensity

struct DropIndicesBased end
struct HMCBased end

Base.@kwdef struct CVIProjection{CVIPE, CVIPO} <: ReactiveMP.AbstractApproximationMethod 
    projection_essentials::CVIPE = CVIProjectionEssentials()
    projection_optional::CVIPO   = CVIProjectionOptional()
end

getcviprojectionessentials(cvi::CVIProjection) = cvi.projection_essentials
getcviprojectionoptional(cvi::CVIProjection) = cvi.projection_optional
getcviprojectionconditioners(cvi::CVIProjection) = getcviprojectionconditioners(getcviprojectionessentials(cvi))
getcviprojectiontypes(cvi::CVIProjection) = getcviprojectiontypes(getcviprojectionessentials(cvi))
getcviprojectiondims(cvi::CVIProjection) = getcviprojectiondims(getcviprojectionessentials(cvi))
getcviprojectionparameters(cvi::CVIProjection) = getcviprojectionparameters(getcviprojectionessentials(cvi))
getcvioutsamplesno(cvi::CVIProjection) = getcvioutsamplesno(getcviprojectionoptional(cvi))
getcvimarginalsamplesno(t::T, cvi::CVIProjection) where {T} = getcvimarginalsamplesno(t,getcviprojectionoptional(cvi))
getcvirng(cvi::CVIProjection) = getcvirng(getcviprojectionoptional(cvi))
getcviinitialsamples(cvi::CVIProjection) = getcviinitialsamples(getcviprojectionessentials(cvi))
getcviinitialnaturalparameters(cvi::CVIProjection) = getcviinitialnaturalparameters(getcviprojectionessentials(cvi))


Base.@kwdef struct CVIProjectionEssentials{TS, DS, IS, INP, CS, P}
    projection_types::TS = (out = nothing, in = nothing)
    projection_dims::DS = (out = nothing, in = nothing)
    initial_samples::IS = (out = nothing, in = nothing)
    initial_naturalparameters::INP = (out = nothing, in = nothing)
    projection_conditioners::CS = (out = nothing, in = nothing)
    projection_parameters::P = ExponentialFamilyProjection.DefaultProjectionParameters()
end

getcviprojectionconditioners(cvipe::CVIProjectionEssentials) = cvipe.projection_conditioners
getcviprojectiontypes(cvipe::CVIProjectionEssentials) = cvipe.projection_types
getcviprojectiondims(cvipe::CVIProjectionEssentials) = cvipe.projection_dims
getcviprojectionparameters(cvipe::CVIProjectionEssentials) = cvipe.projection_parameters
getcviinitialsamples(cvipe::CVIProjectionEssentials) = cvipe.initial_samples
getcviinitialnaturalparameters(cvipe::CVIProjectionEssentials) = cvipe.initial_naturalparameters

Base.@kwdef struct CVIProjectionOptional{OS, MS, R} 
    out_samples_no::OS = 1000
    marginal_samples_no::MS = (1000, 5)
    rng::R = Random.MersenneTwister(42)
end
getcvioutsamplesno(cvipo::CVIProjectionOptional) = cvipo.out_samples_no
getcvimarginalsamplesno(::HMCBased, cvipo::CVIProjectionOptional) = first(cvipo.marginal_samples_no)
getcvimarginalsamplesno(::DropIndicesBased, cvipo::CVIProjectionOptional) = last(cvipo.marginal_samples_no)
getcvirng(cvipo::CVIProjectionOptional) = cvipo.rng

struct DivisionOf{A, B}
    numerator::A
    denumerator::B
end

BayesBase.insupport(d::DivisionOf, p) = insupport(d.numerator, p) && insupport(d.denumerator, p)
BayesBase.logpdf(d::DivisionOf, p) = logpdf(d.numerator, p) - logpdf(d.denumerator, p)

function (DO::DivisionOf)(x)
    return logpdf(DO, x)
end


# cost function
function targetfn(M, p, data)
    ef = convert(ExponentialFamilyDistribution, M, p)
    return -mean(logpdf(ef, data))
end

function grad_targetfn(M, p, data)
    ef = convert(ExponentialFamilyDistribution, M, p)
    invfisher = cholinv(Hermitian(fisherinformation(ef)))
    X = ExponentialFamilyProjection.ExponentialFamilyManifolds.partition_point(M, invfisher*ForwardDiff.gradient((p) -> targetfn(M, p, data),p))
    return ExponentialFamilyProjection.Manopt.project(M, p, X)
end

struct LogTargetDensity{I, F}
    dim :: I
    μ   :: F
end

LogDensityProblems.logdensity(p::LogTargetDensity, x) = p.μ(x)
LogDensityProblems.capabilities(::LogTargetDensity) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(p::LogTargetDensity)   = p.dim

function log_target_adjusted_log_pdf(::Type{Univariate}, m_in::Union{Distribution, ExponentialFamilyDistribution, DivisionOf}, _)
    return x -> logpdf(m_in, first(x))
end
function log_target_adjusted_log_pdf(::Type{Multivariate}, m_in::Union{Distribution, ExponentialFamilyDistribution, DivisionOf}, _)
    return x -> logpdf(m_in, x)
end
function log_target_adjusted_log_pdf(::Type{Matrixvariate}, m_in::Union{Distribution, ExponentialFamilyDistribution, DivisionOf}, dims)
    return x -> logpdf(m_in, reshape(x,dims))
end
function log_target_adjusted_log_pdf(::Type{Univariate}, logmeasure, _)
    return x -> logmeasure(first(x))
end
log_target_adjusted_log_pdf(::Type{Multivariate}, logmeasure, _) = logmeasure
function log_target_adjusted_log_pdf(::Type{Matrixvariate}, logmeasure, dims)
    return x -> logmeasure(reshape(x,dims))
end

function hmc_samples(rng, d, log_target_density, initial_x; no_samples = 2_000, n_adapts = 1_000, acceptance_probability = 0.8)
    metric = AdvancedHMC.DiagEuclideanMetric(d) ### We should use fisher metric here
    hamiltonian = AdvancedHMC.Hamiltonian(metric, log_target_density, ForwardDiff)
    initial_ϵ = AdvancedHMC.find_good_stepsize(hamiltonian, initial_x)
    integrator = AdvancedHMC.Leapfrog(initial_ϵ)
    
    kernel     = AdvancedHMC.HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
    adaptor    = AdvancedHMC.StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(acceptance_probability, integrator))
    samples, _ = AdvancedHMC.sample(rng, hamiltonian, kernel, initial_x, no_samples+1, adaptor, n_adapts; verbose = false, progress=false)

    return samples[2:end]
end

vectorize_sample(::Type{Univariate}, sample) = [sample]
vectorize_sample(::Type{Multivariate}, sample) = sample
vectorize_sample(::Type{Matrixvariate}, sample) = vec(sample)

vectorized_rand_with_variate_type(::Type{Univariate}, rng, m_in) = [rand(rng, m_in)]
vectorized_rand_with_variate_type(::Type{Multivariate}, rng, m_in) = rand(rng, m_in)
vectorized_rand_with_variate_type(::Type{Matrixvariate}, rng, m_in)  = vec(rand(rng, m_in))

modify_vectorized_samples_with_variate_type(::Type{Univariate}, samples, _) = map(sample ->first(sample) ,samples)
modify_vectorized_samples_with_variate_type(::Type{Multivariate}, samples,_) = samples
modify_vectorized_samples_with_variate_type(::Type{Matrixvariate}, samples,dims) = map(sample -> reshape(sample, dims), samples)

initialize_cvi_samples(method, rng, m_in, k, symbol) = !isnothing(getcviinitialsamples(method)[symbol]) ? getindex(getcviinitialsamples(method)[symbol], k) : rand(rng, m_in) 
initialize_cvi_natural_parameters(method, rng, manifold, k, symbol) = !isnothing(getcviinitialnaturalparameters(method)[symbol]) ? ExponentialFamilyProjection.partition_point(manifold,getindex(getcviinitialnaturalparameters(method)[symbol], k)) : rand(rng, manifold)

function BayesBase.prod(::GenericProd, something, division::DivisionOf) 
    return prod(GenericProd(), division, something)
end

function BayesBase.prod(::GenericProd, division::DivisionOf, something)
    if division.denumerator == something
        return division.numerator
    else
        return ProductOf(division, something)
    end
end

BayesBase.prod(::GenericProd, division_left::DivisionOf, division_right::DivisionOf) = ProductOf(division_left, division_right)

function __auxiliary_variables(rng, m_ins, method, N)

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

    return var_form_ins, dims_in, sum_dim_in, start_indices, manifolds, natural_parameters_efs, initial_sample

end


include("layout/cvi_projection.jl")
include("rules/in.jl")
include("rules/out.jl")
include("rules/marginals.jl")

# This will enable the extension and make `CVIProjection` compatible with delta nodes 
# Otherwise it should throw an error suggesting users to install `ExponentialFamilyProjection`
# See `approximations/cvi_projection.jl`
ReactiveMP.is_delta_node_compatible(::CVIProjection) = Val(true)

end