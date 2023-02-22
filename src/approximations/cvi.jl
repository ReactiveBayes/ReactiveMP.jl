export CVI, ProdCVI, ForwardDiffGrad

using Random
using LinearAlgebra

"""
    cvi_update!(opt, λ, ∇)
"""
function cvi_update! end

function cvi_update!(callback::F, λ, ∇) where {F <: Function}
    return callback(λ, ∇)
end

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
    Run `using Flux` in your Julia session to enable the `Flux` optimizers support for the CVI approximation method.
    Run `using Zygote` in your Julia session to enable the `ZygoteGrad()` option support for the CVI `grad` parameter.
    Run `using DiffResults` in your Julia session to enable faster gradient computations in case if all inputs are of the `Gaussian` type.
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

struct ForwardDiffGrad end

compute_derivative(::ForwardDiffGrad, f::F, value::Real) where {F}       = ForwardDiff.derivative(f, value)
compute_gradient(::ForwardDiffGrad, f::F, vec::AbstractVector) where {F} = ForwardDiff.gradient(f, vec)
compute_hessian(::ForwardDiffGrad, f::F, vec::AbstractVector) where {F}  = ForwardDiff.hessian(f, vec)

function compute_second_derivative(grad::G, logp::F, z_s::Real) where {G, F}
    first_derivative = (x) -> compute_derivative(grad, logp, x)
    return compute_derivative(grad, first_derivative, z_s)
end

# We perform the check in case if the `enforce_proper_messages` setting is set to `Val{true}`
enforce_proper_message(::Val{true}, λ::NaturalParameters, η::NaturalParameters) = isproper(λ - η)

# We skip the check in case if the `enforce_proper_messages` setting is set to `Val{false}`
enforce_proper_message(::Val{false}, λ::NaturalParameters, η::NaturalParameters) = true

function compute_fisher_matrix(approximation::CVI, ::Type{T}, vec::AbstractVector) where {T <: NaturalParameters}
    neg_lognormalizer = (x) -> -lognormalizer(as_naturalparams(T, x))

    return -compute_hessian(approximation.grad, neg_lognormalizer, vec)
end

function prod(approximation::CVI, outbound, inbound)
    rng = something(approximation.rng, Random.default_rng())

    # Natural parameters of incoming distribution message
    η_inbound = naturalparams(inbound)

    # Natural parameter type of incoming distribution
    T = typeof(η_inbound)

    # Initial parameters of projected distribution
    λ_current = naturalparams(inbound)

    hasupdated = false

    for _ in 1:(approximation.n_iterations)

        # Some distributions implement "sampling" efficient versions
        # returns the same distribution by default
        _, q_friendly = logpdf_sample_friendly(convert(Distribution, λ_current))

        samples = cvilinearize(rand(rng, q_friendly, approximation.n_gradpoints))

        # compute gradient of log-likelihood
        # the multiplication between two logpdfs is correct
        # we take the derivative with respect to `η`
        # `logpdf(outbound, sample)` does not depend on `η` and is just a simple scalar constant
        # see the method papers for more details
        # - https://arxiv.org/pdf/1401.0118.pdf
        # - https://doi.org/10.1016/j.ijar.2022.06.006
        logq = let samples = samples, outbound = outbound, T = T
            (η) -> mean((sample) -> logpdf(outbound, sample) * logpdf(as_naturalparams(T, η), sample), samples)
        end

        ∇logq = compute_gradient(approximation.grad, logq, vec(λ_current))

        # compute Fisher matrix 
        Fisher = compute_fisher_matrix(approximation, T, vec(λ_current))

        # compute natural gradient
        ∇f = Fisher \ ∇logq

        # compute gradient on natural parameters
        ∇ = λ_current - η_inbound - as_naturalparams(T, ∇f)

        # perform gradient descent step
        λ_new = as_naturalparams(T, cvi_update!(approximation.opt, λ_current, ∇))

        # check whether updated natural parameters are proper
        if isproper(λ_new) && enforce_proper_message(approximation.enforce_proper_messages, λ_new, η_inbound)
            λ_current = λ_new
            hasupdated = true
        end
    end

    if !hasupdated && approximation.warn
        @warn "CVI approximation has not updated the initial state. The method did not converge. Set `warn = false` to supress this warning."
    end

    return convert(Distribution, λ_current)
end
