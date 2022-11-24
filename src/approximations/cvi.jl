export CVI, ForwardDiffGrad

using Random

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

The `ProdCVI` structure defines the approximation method of the `Delta` factor node.
This method performs an approximation of the messages through the `Delta` factor node with Stochastic Variational message passing (SVMP-CVI) (See [`Probabilistic programming with stochastic variational message passing`](https://biaslab.github.io/publication/probabilistic_programming_with_stochastic_variational_message_passing/)).

Arguments
 - `rng`: random number generator
 - `n_samples`: number of samples to use for statistics approximation
 - `num_iterations`: number of iteration for the natural parameters gradient optimization
 - `opt`: optimizer, which will be used to perform the natural parameters gradient optimization step
 - `grad`: optional, default to `ForwardDiffGrad`, structure to select how the gradient and the hessian will be computed
 - `warn`: optional, defaults to false, enables or disables warnings related to the optimization steps
 - `enforce_proper_messages`: optional, defaults to true, ensures that a message, computed towards the inbound edges, is a proper distribution

!!! note 
    Run `using Flux` in your Julia session to enable the `Flux` optimizers support for the CVI approximation method.

!!! note 
    Run `using Zygote` in your Julia session to enable the `ZygoteGrad` option support for the CVI `grad` parameter.

"""
struct ProdCVI{R, O, G} <: AbstractApproximationMethod
    rng::R
    n_samples::Int
    num_iterations::Int
    opt::O
    grad::G
    warn::Bool
    enforce_proper_messages::Bool
end

get_grad(approximation::ProdCVI) = approximation.grad

function ProdCVI(rng::AbstractRNG, n_samples::Int, num_iterations::Int, opt::O) where {O}
    return ProdCVI(rng, n_samples, num_iterations, opt, ForwardDiffGrad(), false, true)
end

function ProdCVI(rng::AbstractRNG, n_samples::Int, num_iterations::Int, opt::O, grad::G) where {O, G}
    return ProdCVI(rng, n_samples, num_iterations, opt, grad, false, true)
end

function ProdCVI(n_samples::Int, num_iterations::Int, opt::O, warn::Bool = false) where {O}
    return ProdCVI(Random.GLOBAL_RNG, n_samples, num_iterations, opt, ForwardDiffGrad(), warn, true)
end

function ProdCVI(n_samples::Int, num_iterations::Int, opt::O, grad::G, warn::Bool = false) where {O, G}
    return ProdCVI(Random.GLOBAL_RNG, n_samples, num_iterations, opt, grad, warn, true)
end

"""
Alias for the `ProdCVI` method.

See also: [`ProdCVI`](@ref)
"""
const CVI = ProdCVI

#---------------------------
# CVI implementations
#---------------------------

struct ForwardDiffGrad end

function compute_derivative(::ForwardDiffGrad, f::F, param::Real) where {F}
    ForwardDiff.derivative(f, param)
end

function compute_gradient(::ForwardDiffGrad, f::F, vec_params) where {F}
    ForwardDiff.gradient(f, vec_params)
end

function compute_hessian(::ForwardDiffGrad, f::F, vec_params) where {F}
    ForwardDiff.hessian(f, vec_params)
end

function enforce_proper_message(enforce::Bool, λ::NaturalParameters, η::NaturalParameters)
    return !enforce || (enforce && isproper(λ - η))
end

function compute_fisher_matrix(approximation::CVI, ::Type{T}, params::Vector) where {T <: NaturalParameters}

    # specify lognormalizer function
    lognormalizer_function = (x) -> lognormalizer(as_naturalparams(T, x))

    #  compute Fisher matrix
    F = compute_hessian(get_grad(approximation), lognormalizer_function, params)

    #  return Fisher matrix
    return F
end

# without type constraints it will create stakeoverflow error
# prod(approximation::CVI, dist, logp::F) where {F} = prod(approximation, logp, dist)

function prod(approximation::CVI, logp::F, dist) where {F <: Function}
    rng = something(approximation.rng, Random.GLOBAL_RNG)

    #  natural parameters of incoming distribution message
    η = naturalparams(dist)

    # Natural parameter type of incoming distribution
    T = typeof(η)

    # initial parameters of projected distribution
    λ = naturalparams(dist)

    # initialize update flag
    hasupdated = false

    for _ in 1:(approximation.num_iterations)

        # create distribution to sample from and sample from it
        q = convert(Distribution, λ)
        _, q_friendly = logpdf_sample_friendly(q)
        z_s = rand(rng, q_friendly)

        # compute gradient of log-likelihood
        logq = (x) -> logpdf(as_naturalparams(T, x), z_s)
        ∇logq = logp(z_s) .* compute_gradient(get_grad(approximation), logq, vec(λ))

        # compute Fisher matrix and Cholesky decomposition
        Fisher = compute_fisher_matrix(approximation, T, vec(λ))
        F_chol = fastcholesky!(Fisher)

        # compute natural gradient
        ∇f = F_chol \ ∇logq

        # compute gradient on natural parameters
        ∇ = λ - η - as_naturalparams(T, ∇f)

        # perform gradient descent step
        λ_new = as_naturalparams(T, cvi_update!(approximation.opt, λ, ∇))

        # check whether updated natural parameters are proper
        if isproper(λ_new) && enforce_proper_message(approximation.enforce_proper_messages, λ_new, η)
            λ = λ_new
            hasupdated = true
        end
    end

    if !hasupdated && approximation.warn
        @warn "CVI approximation has not updated the initial state. The method did not converge. Set `warn = false` to supress this warning."
    end

    return λ
end
