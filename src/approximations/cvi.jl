export CVI, ProdCVI
export ZygoteGrad, ForwardDiffGrad

using Random
using LinearAlgebra
using DiffResults

"""
    cvi_setup!(opt, λ)

Initialises the given optimiser for the CVI procedure given the structure of λ.
Returns a tuple of the optimiser and the optimiser state.
"""
function cvi_setup! end

"""
    cvi_update!(tuple_of_opt_and_state, new_λ, λ, ∇)

Uses the optimiser, its state and the gradient ∇ to change the trainable parameters in the λ.
Modifies the optimiser state and and store the output in the new_λ. Returns a tuple of the optimiser and the new_λ.
"""
function cvi_update! end

# Specialized method for callback functions, a user can provide an arbitrary callback with the optimization procedure
cvi_setup(callback::F, λ) where {F <: Function} = (callback, nothing)
cvi_update!(tuple::Tuple{F, Nothing}, new_λ, λ, ∇) where {F <: Function} = (tuple, tuple[1](new_λ, λ, ∇))

cvilinearize(vector::AbstractVector) = vector
cvilinearize(matrix::AbstractMatrix) = eachcol(matrix)

"""
    ProdCVI

The `ProdCVI` structure defines the approximation method hyperparameters of the `prod(approximation::CVI, logp::F, dist)`.
This method performs an approximation of the product of the `dist` and `logp` with Stochastic Variational message passing (SVMP-CVI) (See [`Probabilistic programming with stochastic variational message passing`](https://biaslab.github.io/publication/probabilistic_programming_with_stochastic_variational_message_passing/)).

Arguments
 - `rng`: random number generator
 - `n_samples`: number of samples to use for statistics approximation
 - `n_iterations`: number of iteration for the natural parameters gradient optimization
 - `opt`: optimizer, which will be used to perform the natural parameters gradient optimization step
 - `grad`: optional, defaults to `ForwardDiffGrad()`, structure to select how the gradient and the hessian will be computed
 - `n_gradpoints`: optional, defaults to 1, number of points to estimate gradient of the likelihood (dist*logp)
 - `enforce_proper_messages`: optional, defaults to true, ensures that a message, computed towards the inbound edges, is a proper distribution, must be of type `Val(true)/Val(false)`
 - `warn`: optional, defaults to true, enables or disables warnings related to the optimization steps

!!! note 
    `n_gradpoints` option is ignored in the Gaussian case

!!! note 
    Adding the `Optimisers.jl` in your Julia environment enables additional optimizers from the `Optimisers.jl` for the CVI approximation method.
    Adding the `Zygote` in your Julia environment enables the `ZygoteGrad()` option for the CVI `grad` parameter.
    Adding the `DiffResults` in your Julia environment enables faster gradient computations in case if all inputs are of the `Gaussian` type.
"""
struct ProdCVI{R, O, G, B} <: AbstractApproximationMethod
    rng::R
    n_samples::Int
    n_iterations::Int
    opt::O
    grad::G
    n_gradpoints::Int
    enforce_proper_messages::Val{B}
    warn::Bool

    function ProdCVI(
        rng::R, n_samples::Int, n_iterations::Int, opt::O, grad::G = ForwardDiffGrad(), n_gradpoints::Int = 1, enforce_proper_messages::Val{B} = Val(true), warn::Bool = true
    ) where {R, O, G, B}
        return new{R, O, G, B}(rng, n_samples, n_iterations, opt, grad, n_gradpoints, enforce_proper_messages, warn)
    end
end

function ProdCVI(n_samples::Int, n_iterations::Int, opt, grad = ForwardDiffGrad(), n_gradpoints::Int = 1, enforce_proper_messages::Val = Val(true), warn::Bool = true)
    return ProdCVI(Random.GLOBAL_RNG, n_samples, n_iterations, opt, grad, n_gradpoints, enforce_proper_messages, warn)
end

"""Alias for the `ProdCVI` method. See help for [`ProdCVI`](@ref)"""
const CVI = ProdCVI

#---------------------------
# CVI implementations
#---------------------------

"""
    ForwardDiffGrad(chunk_size::Int)

The auto-differentiation backend for the CVI procedure.
Uses the `ForwardDiff` library to compute gradients/derivatives.
If `chunk_size` is not specified then uses the heuristic from `ForwardDiff`, which is type-unstable.

!!! note
    The `ForwardDiff.jl` must be added to the current Julia environment.
"""
struct ForwardDiffGrad{N} end

ForwardDiffGrad() = ForwardDiffGrad{0}()
ForwardDiffGrad(N::Int) = ForwardDiffGrad{N}()

# Chunks may improve the performance, but should be specified carefully basis by basis
getchunk(::ForwardDiffGrad{0}, η::AbstractVector) = ForwardDiff.Chunk(η)
getchunk(::ForwardDiffGrad{0}, η::Number) = ForwardDiff.Chunk{1}()
getchunk(::ForwardDiffGrad{N}, _) where {N} = ForwardDiff.Chunk{N}()

function compute_second_derivative(grad::G, logp::F, z_s::Real) where {G, F}
    first_derivative = (x) -> compute_derivative(grad, logp, x)
    return compute_derivative(grad, first_derivative, z_s)
end

function compute_derivative(::ForwardDiffGrad, f::F, value::T)::T where {F, T}
    return ForwardDiff.derivative(f, value)
end

function compute_gradient!(::ForwardDiffGrad, result, cfg, f::F, vec::AbstractVector{T})::Vector{T} where {F, T}
    ForwardDiff.gradient!(result, f, vec, cfg)
    return DiffResults.gradient(result)
end

function compute_hessian!(::ForwardDiffGrad, result, cfg, f::F, vec::AbstractVector{T})::Matrix{T} where {F, T}
    ForwardDiff.hessian!(result, f, vec, cfg)
    return DiffResults.hessian(result)
end

# We perform the check in case if the `enforce_proper_messages` setting is set to `Val{true}`
function enforce_proper_message(::Val{true}, ::Type{T}, cache, λ, η, conditioner) where {T}
    # cache = λ .- η
    @inbounds for (i, λᵢ, ηᵢ) in zip(eachindex(cache), λ, η)
        cache[i] = λᵢ - ηᵢ
    end
    return isproper(NaturalParametersSpace(), T, cache, conditioner)
end

# We skip the check in case if the `enforce_proper_messages` setting is set to `Val{false}`
enforce_proper_message(::Val{false}, ::Type{T}, cache, λ, η, conditioner) where {T} = true

# We need this structure to aboid performance issues with type parameter `T` in lambda functions
# Otherwise the invokation of such function would be about 10x slower
struct LogGradientInvoker{T, S, O, C}
    samples::S
    outbound::O
    conditioner::C

    function LogGradientInvoker(::Type{T}, samples::S, outbound::O, conditioner::C) where {T, S, O, C}
        return new{T, S, O, C}(samples, outbound, conditioner)
    end
end

# Some methods for some distributions require extra cache to be preallocated
# Look for the the optimized versions in the bottom of this file
function prepare_log_gradient_invoker_cache(::Type{T}, grad::ForwardDiffGrad, η, f) where {T}
    ad_cache = DiffResults.DiffResult(first(η), similar(η), similar(η, length(η), length(η)))
    chunk = getchunk(grad, η)
    ad_cfgs = (ForwardDiff.GradientConfig(f, η, chunk), ForwardDiff.HessianConfig(f, ad_cache, η, chunk))
    ∇logq = similar(η)
    ∇f = similar(η)
    return (ad_cache, ad_cfgs, ∇logq, ∇f)
end

# compute gradient of log-likelihood
# the multiplication between two logpdfs is correct
# we take the derivative with respect to `η`
# `logpdf(outbound, sample)` does not depend on `η` and is just a simple scalar constant
# see the method papers for more details
# - https://arxiv.org/pdf/1401.0118.pdf
# - https://doi.org/10.1016/j.ijar.2022.06.006
function (invoker::LogGradientInvoker{T})(η) where {T}
    return mean(invoker.samples) do sample
        return logpdf(invoker.outbound, sample) * logpdf(ExponentialFamilyDistribution(T, η, invoker.conditioner, nothing), sample)
    end
end

# Look for the the optimized versions in the bottom of this file
function estimate_natural_gradient!(grad::ForwardDiffGrad, cache, invoker::LogGradientInvoker, current)
    (ad_cache, ad_cfgs, ∇logq, ∇f) = cache

    point = getnaturalparameters(current)
    ∇logq = compute_gradient!(grad, ad_cache, ad_cfgs[1], invoker, point)

    # compute Fisher matrix 
    Fisher = fisherinformation(current)

    # compute natural gradient
    # cholinv(Fisher) * ∇logq
    return mul!(∇f, cholinv(Fisher), ∇logq)::typeof(∇f)
end

function prod(approximation::CVI, outbound, inbound)
    rng = something(approximation.rng, Random.default_rng())

    # Natural parameters of incoming distribution message
    inbound_ef = convert(ExponentialFamilyDistribution, inbound)
    inbound_η = getnaturalparameters(inbound_ef)
    inbound_c = getconditioner(inbound_ef)

    # Optimizer procedure may depend on the type of the inbound natural parameters
    optimizer_and_state = cvi_setup(approximation.opt, inbound_η)
    gradmethod = approximation.grad

    # Natural parameter type of incoming distribution
    T = ExponentialFamily.exponential_family_typetag(inbound_ef)

    # Initial parameters of projected distribution
    current_ef = convert(ExponentialFamilyDistribution, inbound) # current EF distribution
    current_λ  = getnaturalparameters(current_ef) # current natural parameters
    scontainer = rand(rng, sampling_optimized(inbound), approximation.n_gradpoints) # sampling container
    current_∇  = similar(current_λ) # current gradient
    new_λ      = similar(current_λ) # new natural parameters
    cache      = similar(current_λ) # just intermediate buffer

    # We avoid use of lambda functions, because they cannot capture `T`
    # which leads to performance issues 
    # + some types `T` implement a more accure and efficient estimater
    invoker = LogGradientInvoker(T, cvilinearize(scontainer), outbound, inbound_c)
    logqcache = prepare_log_gradient_invoker_cache(T, gradmethod, current_λ, invoker)

    hasupdated = false

    for _ in 1:(approximation.n_iterations)

        # Some distributions implement "sampling" efficient versions
        # returns the same distribution by default
        samples = cvilinearize(rand!(rng, sampling_optimized(convert(Distribution, current_ef)), scontainer))

        # compute gradient of log-likelihood
        ∇f = estimate_natural_gradient!(gradmethod, logqcache, invoker, current_ef)

        # compute gradient on natural parameters (current_∇ = current_λ .- inbound_η .- ∇f)
        @inbounds for (i, λᵢ, ηᵢ, ∇fᵢ) in zip(eachindex(current_∇), current_λ, inbound_η, ∇f)
            current_∇[i] = λᵢ - ηᵢ - ∇fᵢ
        end

        # perform gradient descent step
        optimizer_and_state, new_λ = cvi_update!(optimizer_and_state, new_λ, current_λ, current_∇)

        # check whether updated natural parameters are proper
        if isproper(NaturalParametersSpace(), T, new_λ, inbound_c) && enforce_proper_message(approximation.enforce_proper_messages, T, cache, new_λ, inbound_η, inbound_c)
            copyto!(current_λ, new_λ)
            hasupdated = true
        end
    end

    if !hasupdated && approximation.warn
        @warn "CVI approximation has not updated the initial state. The method did not converge. Set `warn = false` to supress this warning."
    end

    return convert(Distribution, current_ef)
end

# Thes functions extends the `CVI` approximation method in case if input is from the `NormalDistributionsFamily`

function compute_df_mv(grad::ForwardDiffGrad, _, logp::F, z_s::Real) where {F}
    df_m = compute_derivative(grad, logp, z_s)
    df_v = compute_second_derivative(grad, logp, z_s)
    return df_m, df_v / 2
end

function compute_df_mv(grad::ForwardDiffGrad, cache, logp::F, z_s::AbstractVector) where {F}
    # Extract cache and configs for the ForwardDiff
    ad_cache, ad_cfgs, _, _ = cache
    # Compute the hessian in-place
    compute_hessian!(grad, ad_cache, ad_cfgs[2], logp, z_s)
    # Extract the gradient and the hessian from the resulting `ad_cache`
    df_m = DiffResults.gradient(ad_cache)
    df_v = DiffResults.hessian(ad_cache)
    # inplace df_v ./ 2
    map!(Base.Fix2(/, 2), df_v, df_v)
    return df_m, df_v
end

function prepare_log_gradient_invoker_cache(::Type{T}, grad::ForwardDiffGrad, η, invoker) where {T <: NormalDistributionsFamily}
    # Specialized version for gaussians takes gradients and hessians with respect to a different function `f`
    f = (x) -> logpdf(invoker.outbound, x)
    ad_cache, ad_cfgs = __gausian_ad_cache(grad, first(invoker.samples), f)
    n = convert(Int, (-1 + sqrt(4 * length(η) + 1)) / 2)
    ∇f = similar(η)     # Stores the actual estimated natural gradient
    tmp = similar(η, n) # Intermediate storage for buffer calculations
    return (ad_cache, ad_cfgs, f, ∇f, tmp)
end

function __gaussian_ad_cache(grad::ForwardDiffGrad, sample::Number, f)
    # We do not implement any cache for the univariate Gaussian case
    return nothing, nothing
end

function __gaussian_ad_cache(grad::ForwardDiffGrad, sample::AbstractArray, f)
    # We prepare a different `DiffResult` container specialized for `f` specifically
    R = eltype(sample)
    k = length(sample)
    chunk = getchunk(grad, sample)
    # The hessians and gradients will be store in the `ad_cache` later on
    ad_cache = DiffResults.DiffResult(first(sample), Vector{R}(undef, k), Matrix{R}(undef, k, k))
    # ForwardDiff configs are specialized for `f`
    ad_cfgs = (ForwardDiff.GradientConfig(f, sample, chunk), ForwardDiff.HessianConfig(f, ad_cache, sample, chunk))
    return ad_cache, ad_cfgs
end

# This procedure does not call the `invoker`, 
# but instead has a different target function saved in the `prepare_log_gradient_invoker_cache`
function estimate_natural_gradient!(grad::ForwardDiffGrad, cache, invoker::LogGradientInvoker{T}, current) where {T <: NormalDistributionsFamily}
    μ = mean(current)
    K = length(invoker.samples)
    _, _, f, ∇f, tmp = cache

    fill!(∇f, zero(eltype(μ)))

    # Below is a hand-written and optimized version of the following code:
    # return sum((z_s) -> begin
    #     df_m, df_v = compute_df_mv(grad, (cache, df_m, df_v), f, z_s)
    #     df_μ1 = df_m - 2 * df_v * μ
    #     df_μ2 = df_v
    #     ExponentialFamily.pack_parameters(T, (df_μ1 ./ K, df_μ2 ./ K))
    # end, invoker.samples)

    for sample in invoker.samples
        df_m, df_v = compute_df_mv(grad, cache, f, sample)

        if df_m isa Number 
            tmp = df_v * μ
        else
            mul!(tmp, df_v, μ)
        end

        k = firstindex(∇f)
        @inbounds for (df_mᵢ, tmpᵢ) in zip(df_m, tmp)
            ∇f[k] += (df_mᵢ - 2 * tmpᵢ) / K
            k = k + 1
        end

        @inbounds for df_vᵢ in df_v
            ∇f[k] += df_vᵢ / K
            k = k + 1
        end
    end

    return ∇f
end