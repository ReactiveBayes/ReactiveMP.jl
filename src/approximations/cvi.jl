export CVI, ProdCVI
export ZygoteGrad, ForwardDiffGrad

using Random
using LinearAlgebra

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
The auto-differentiation backend for the CVI procedure.
Uses the `Zygote` library to compute gradients/derivatives.

!!! note
    The `Zygote.jl` must be added to the current Julia environment.
"""
struct ZygoteGrad end

"""
The auto-differentiation backend for the CVI procedure.
Uses the `ForwardDiff` library to compute gradients/derivatives.

!!! note
    The `ForwardDiff.jl` must be added to the current Julia environment.
"""
struct ForwardDiffGrad end

function compute_derivative(::ForwardDiffGrad, f::F, value::T)::T where {F, T}
    return ForwardDiff.derivative(f, value)
end

function compute_second_derivative(grad::G, logp::F, z_s::Real) where {G, F}
    first_derivative = (x) -> compute_derivative(grad, logp, x)
    return compute_derivative(grad, first_derivative, z_s)
end

function compute_gradient!(::ForwardDiffGrad, output::Vector{T}, f::F, vec::AbstractVector{T})::Vector{T} where {F, T}
    return ForwardDiff.gradient!(output, f, vec)
end

function compute_hessian!(::ForwardDiffGrad, output::Matrix{T}, f::F, vec::AbstractVector{T})::Matrix{T} where {F, T}
    return ForwardDiff.hessian!(output, f, vec)
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
struct LogGradientInvoker{T, S, O, C, H}
    samples::S
    outbound::O
    conditioner::C
    cache::H

    function LogGradientInvoker(::Type{T}, samples::S, outbound::O, conditioner::C, cache::H) where {T, S, O, C, H}
        return new{T, S, O, C, H}(samples, outbound, conditioner, cache)
    end
end

# Some methods for some distributions require extra cache to be preallocated
# Look for the the optimized versions in the bottom of this file
prepare_log_gradient_invoker_cache(::Type{T}, η) where {T} = (similar(η), similar(η))

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
function estimate_natural_gradient!(invoker::LogGradientInvoker, grad, current)
    (∇logq, ∇f) = invoker.cache

    ∇logq = compute_gradient!(grad, ∇logq, invoker, getnaturalparameters(current))

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

    # Natural parameter type of incoming distribution
    T = ExponentialFamily.exponential_family_typetag(inbound_ef)

    # Initial parameters of projected distribution
    current_ef = convert(ExponentialFamilyDistribution, inbound) # current EF distribution
    current_λ  = getnaturalparameters(current_ef) # current natural parameters
    scontainer = rand(rng, sampling_optimized(inbound), approximation.n_gradpoints) # sampling container
    current_∇  = similar(current_λ) # current gradient
    new_λ      = similar(current_λ) # new natural parameters
    cache      = similar(current_λ) # just intermediate buffer
    logqcache  = prepare_log_gradient_invoker_cache(T, current_λ)

    hasupdated = false

    for _ in 1:(approximation.n_iterations)

        # Some distributions implement "sampling" efficient versions
        # returns the same distribution by default
        samples = cvilinearize(rand!(rng, sampling_optimized(convert(Distribution, current_ef)), scontainer))

        # We avoid use of lambda functions, because they cannot capture `T`
        # which leads to performance issues 
        # + some types `T` implement a more accure and efficient estimater
        invoker = LogGradientInvoker(T, samples, outbound, inbound_c, logqcache)

        # compute gradient of log-likelihood
        ∇f = estimate_natural_gradient!(invoker, approximation.grad, current_ef)

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

function compute_df_mv(grad, _, logp::F, z_s::Real) where {F}
    df_m = compute_derivative(grad, logp, z_s)
    df_v = compute_second_derivative(grad, logp, z_s)
    return df_m, df_v / 2
end

function compute_df_mv(grad, cache, logp::F, z_s::AbstractVector) where {F}
    df_m, df_v = cache
    df_m = compute_gradient!(grad, df_m, logp, z_s)
    df_v = compute_hessian!(grad, df_v, logp, z_s)
    return df_m, df_v ./ 2
end

# Specialized version for the Gaussians distribution

function prepare_log_gradient_invoker_cache(::Type{T}, η) where {T <: NormalDistributionsFamily}
    n = convert(Int, (-1 + sqrt(4 * length(η) + 1)) / 2)
    return (similar(η, n), similar(η, (n, n)))
end

function estimate_natural_gradient!(invoker::LogGradientInvoker{T}, grad, current) where {T <: NormalDistributionsFamily}
    μ = mean(current)
    K = length(invoker.samples)
    return sum((z_s) -> begin
        df_m, df_v = compute_df_mv(grad, invoker.cache, (x) -> logpdf(invoker.outbound, x), z_s)
        df_μ1 = df_m - 2 * df_v * μ
        df_μ2 = df_v
        ExponentialFamily.pack_parameters(T, (df_μ1 ./ K, df_μ2 ./ K))
    end, invoker.samples)
end