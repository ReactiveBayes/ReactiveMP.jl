module ReactiveMPProjectionExt

using ReactiveMP, ExponentialFamily, AdvancedHMC, LogDensityProblems, Distributions, ExponentialFamilyProjection, BayesBase, Random, LinearAlgebra, FastCholesky
using ForwardDiff
export CVIProjection,CVIProjectionEssentials,CVIProjectionOptional, DivisionOf, LogTargetDensity
export getcviprojectionessentials,getcviprojectionoptional,getcviprojectionconditioners
export getcviprojectiontypes, getcviprojectionparameters, getcviprojectionparameters,getcvioutsamplesno
export getcvimarginalsamplesno, getcvirng

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
getcvimarginalsamplesno(cvi::CVIProjection) = getcvimarginalsamplesno(getcviprojectionoptional(cvi))
getcvirng(cvi::CVIProjection) = getcvirng(getcviprojectionoptional(cvi))
getcviinitialsamples(cvi::CVIProjection) = getcviinitialsamples(getcviprojectionessentials(cvi))
getcviinitialnaturalparameters(cvi::CVIProjection) = getcviinitialnaturalparameters(getcviprojectionessentials(cvi))


Base.@kwdef struct CVIProjectionEssentials{TS, DS, IS, INP, CS, P}
    projection_types::TS = nothing
    projection_dims::DS = nothing
    initial_samples::IS = nothing
    initial_naturalparameters::INP = nothing
    projection_conditioners::CS = nothing
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
    marginal_samples_no::MS = 1000
    rng::R = Random.MersenneTwister(42)
end
getcvioutsamplesno(cvipo::CVIProjectionOptional) = cvipo.out_samples_no
getcvimarginalsamplesno(cvipo::CVIProjectionOptional) = cvipo.marginal_samples_no
getcvirng(cvipo::CVIProjectionOptional) = cvipo.rng


struct DivisionOf{A, B}
    numerator::A
    denumerator::B
end

BayesBase.insupport(d::DivisionOf, p) = insupport(d.numerator, p) && insupport(d.denumerator, p)
BayesBase.logpdf(d::DivisionOf, p) = logpdf(d.numerator, p) - logpdf(d.denumerator, p)


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


function log_target_adjusted_log_pdf(::Type{Univariate}, m_in, _)
    return x -> logpdf(m_in, first(x))
end

function log_target_adjusted_log_pdf(::Type{Multivariate}, m_in, _)
    return x -> logpdf(m_in, x)
end

function log_target_adjusted_log_pdf(::Type{Matrixvariate}, m_in, dims)
    return x -> logpdf(m_in, reshape(x,dims))
end

function hmc_samples(rng, d, log_target_density, initial_x; no_samples = 2_000, n_adapts = 1_000, acceptance_probability = 0.8)
    metric = AdvancedHMC.DiagEuclideanMetric(d) ### We should use fisher metric here
    hamiltonian = AdvancedHMC.Hamiltonian(metric, log_target_density, ForwardDiff)
    initial_ϵ = AdvancedHMC.find_good_stepsize(hamiltonian, initial_x)
    integrator = AdvancedHMC.Leapfrog(initial_ϵ)
    
    kernel = AdvancedHMC.HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
    adaptor = AdvancedHMC.StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(acceptance_probability, integrator))
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

initialize_cvi_samples(method, rng, m_in, k) = !isnothing(getcviinitialsamples(method)) ? getindex(getcviinitialsamples(method), k) : rand(rng, m_in) 
initialize_cvi_natural_parameters(method, rng, manifold) = !isnothing(getcviinitialnaturalparameters(method)) ? first(getcviinitialnaturalparameters(method)) : rand(rng, manifold)



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

include("layout/cvi_projection.jl")
include("rules/in.jl")
include("rules/out.jl")
include("rules/marginals.jl")

# This will enable the extension and make `CVIProjection` compatible with delta nodes 
# Otherwise it should throw an error suggesting users to install `ExponentialFamilyProjection`
# See `approximations/cvi_projection.jl`
ReactiveMP.is_delta_node_compatible(::ReactiveMP.CVIProjection) = Val(true)

end