export CVIApproximation, CVI, ForwardDiffGrad

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
    CVIApproximation

The `CVIApproximation` structure defines the approximation method of the `Delta` factor node.
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
struct CVIApproximation{R, O, G} <: AbstractApproximationMethod
    rng::R
    n_samples::Int
    num_iterations::Int
    opt::O
    grad::G
    warn::Bool
    enforce_proper_messages::Bool
end

get_grad(approximation::CVIApproximation) = approximation.grad

function CVIApproximation(rng::AbstractRNG, n_samples::Int, num_iterations::Int, opt::O) where {O}
    return CVIApproximation(rng, n_samples, num_iterations, opt, ForwardDiffGrad(), false, true)
end

function CVIApproximation(rng::AbstractRNG, n_samples::Int, num_iterations::Int, opt::O, grad::G) where {O, G}
    return CVIApproximation(rng, n_samples, num_iterations, opt, grad, false, true)
end

function CVIApproximation(n_samples::Int, num_iterations::Int, opt::O, warn::Bool = false) where {O}
    return CVIApproximation(Random.GLOBAL_RNG, n_samples, num_iterations, opt, ForwardDiffGrad(), warn, true)
end

"""
Alias for the `CVIApproximation` method.

See also: [`CVIApproximation`](@ref)
"""
const CVI = CVIApproximation

#---------------------------
# CVI implementations
#---------------------------

struct ForwardDiffGrad end

function compute_grad(::ForwardDiffGrad, f::F, vec_params) where {F}
    ForwardDiff.gradient(f, vec_params)
end

function compute_hessian(::ForwardDiffGrad, f::F, vec_params) where {F}
    ForwardDiff.hessian(f, vec_params)
end

function enforce_proper_message(enforce::Bool, λ::NaturalParameters, η::NaturalParameters)
    return !enforce || (enforce && isproper(λ - η))
end

function render_cvi(approximation::CVIApproximation, logp_nc::F, initial) where {F}
    η = naturalparams(initial)
    λ = naturalparams(initial)
    T = typeof(η)

    rng = something(approximation.rng, Random.GLOBAL_RNG)
    opt = approximation.opt
    its = approximation.num_iterations

    hasupdated = false

    A = (vec_params) -> lognormalizer(as_naturalparams(T, vec_params))
    Fisher = (vec_params) -> compute_hessian(get_grad(approximation), A, vec_params)

    for _ in 1:its
        q = convert(Distribution, λ)
        _, q_friendly = logpdf_sample_friendly(q)

        z_s = rand(rng, q_friendly)

        logq = (vec_params) -> logpdf(as_naturalparams(T, vec_params), z_s)
        ∇logq = compute_grad(get_grad(approximation), logq, vec(λ))

        fisher_matrix = Fisher(vec(λ))
        ∇f = cholinv(fisher_matrix) * (logp_nc(z_s) .* ∇logq)
        ∇ = λ - η - as_naturalparams(T, ∇f)
        updated = as_naturalparams(T, cvi_update!(opt, λ, ∇))

        if isproper(updated) && enforce_proper_message(approximation.enforce_proper_messages, updated, η)
            λ = updated
            hasupdated = true
        end
    end

    if !hasupdated && approximation.warn
        @warn "CVI approximation has not updated the initial state. The method did not converge. Set `warn = false` to supress this warning."
    end

    return λ
end
