# Flow's rules: a normal pushed through the model, forwards towards `out` and backwards towards
# `in`. Linearization uses the model's own Jacobians and returns a covariance form for a covariance
# input, a mean-precision form for a precision one; Unscented uses sigma points and returns the
# covariance form.

"""
    FlowLinearization

The type of a [`FlowApproximation`](@ref) whose method is
[`Linearization`](@extref MessagePassingRulesApproximations.Linearization), the `algorithm` the
linearisation rules are declared under.
"""
const FlowLinearization = FlowApproximation{<:AbstractCompiledFlowModel, <:Linearization}

"""
    FlowUnscented

The type of a [`FlowApproximation`](@ref) whose method is
[`Unscented`](@extref MessagePassingRulesApproximations.Unscented), the `algorithm` the unscented
rules are declared under.
"""
const FlowUnscented = FlowApproximation{<:AbstractCompiledFlowModel, <:Unscented}

@define_message_update_rule(
    node = Flow, target = :out, algorithm = FlowLinearization, args = (m[:in]::MvNormalMeanCovariance,),
    body = (algo, args) -> begin
        μ_in, Σ_in = mean_cov(args.m[:in])
        μ_out, J = forward_jacobian(getmodel(algo), μ_in)
        MvNormalMeanCovariance(μ_out, J * Σ_in * J')
    end,
)

"""
    flow_forward_precision(algo::FlowLinearization, m_in) -> MvNormalMeanPrecision

The linearised message towards `out` from a normal `m_in` in precision form, mean-precision or
weighted-mean-precision alike: mean `forward(model, μ)` and precision `Jᵢᵀ Λ Jᵢ`, with `Jᵢ`
the inverse flow's Jacobian at the output's mean `forward(model, μ)`. It is the covariance form's
message, inverted without inverting a matrix.
"""
flow_forward_precision(algo, m_in) = begin
    μ_in, Λ_in = mean_precision(m_in)
    μ_out = forward(getmodel(algo), μ_in)
    Ji = inv_jacobian(getmodel(algo), μ_out)
    MvNormalMeanPrecision(μ_out, Ji' * Λ_in * Ji)
end

@define_message_update_rule(
    node = Flow, target = :out, algorithm = FlowLinearization, args = (m[:in]::MvNormalMeanPrecision,),
    body = (algo, args) -> flow_forward_precision(algo, args.m[:in]),
)

@define_message_update_rule(
    node = Flow, target = :out, algorithm = FlowLinearization, args = (m[:in]::MvNormalWeightedMeanPrecision,),
    body = (algo, args) -> flow_forward_precision(algo, args.m[:in]),
)

@define_message_update_rule(
    node = Flow, target = :in, algorithm = FlowLinearization, args = (m[:out]::MvNormalMeanCovariance,),
    body = (algo, args) -> begin
        μ_out, Σ_out = mean_cov(args.m[:out])
        μ_in, Ji = backward_inv_jacobian(getmodel(algo), μ_out)
        MvNormalMeanCovariance(μ_in, Ji * Σ_out * Ji')
    end,
)

"""
    flow_backward_precision(algo::FlowLinearization, m_out) -> MvNormalMeanPrecision

The linearised message towards `in` from a normal `m_out` in precision form: mean
`backward(model, μ)` and precision `Jᵀ Λ J`, with `J` the flow's Jacobian at the input's mean
`backward(model, μ)`. It is the covariance form's message, inverted without inverting a matrix.
"""
flow_backward_precision(algo, m_out) = begin
    μ_out, Λ_out = mean_precision(m_out)
    μ_in = backward(getmodel(algo), μ_out)
    J = jacobian(getmodel(algo), μ_in)
    MvNormalMeanPrecision(μ_in, J' * Λ_out * J)
end

@define_message_update_rule(
    node = Flow, target = :in, algorithm = FlowLinearization, args = (m[:out]::MvNormalMeanPrecision,),
    body = (algo, args) -> flow_backward_precision(algo, args.m[:out]),
)

@define_message_update_rule(
    node = Flow, target = :in, algorithm = FlowLinearization, args = (m[:out]::MvNormalWeightedMeanPrecision,),
    body = (algo, args) -> flow_backward_precision(algo, args.m[:out]),
)

"""
    flow_unscented(method::Unscented, dim) -> Unscented

The unscented method with its weights for dimension `dim`: `Unscented(dim)` as given, or one built
from the parameters of `Unscented()`.

# Throws

A `DimensionMismatch` when `method` was built for a dimension other than `dim`.
"""
function flow_unscented(method::Unscented, dim)
    if method.e === nothing
        return Unscented(dim; alpha = method.α, beta = method.β, kappa = method.κ)
    end
    L = MessagePassingRulesApproximations.getL(method)
    L == dim || throw(DimensionMismatch("`Unscented($L)` was given for a flow of dimension $dim"))
    return method
end

"""
    flow_unscented_statistics(f, model, method::Unscented, μ, Σ) -> MvNormalMeanCovariance

The normal of `f(model, x)` for `x ~ N(μ, Σ)` by the unscented transform, `f` being
[`FlowMessagePassingRules.forward`](@ref) or [`FlowMessagePassingRules.backward`](@ref). The
sigma points are placed along the rows of the symmetric square root of `(L + λ) Σ`, not along a
Cholesky factor as
[`unscented_statistics`](@extref MessagePassingRulesApproximations.unscented_statistics) places
them: both are exact for an affine flow, and they differ for a nonlinear one.
"""
function flow_unscented_statistics(f, model, method, μ, Σ)
    T = eltype(model)
    approximation = flow_unscented(method, length(μ))
    A = MessagePassingRulesApproximations
    λ, L, Wm, Wc = A.getλ(approximation), A.getL(approximation), A.getWm(approximation), A.getWc(approximation)
    sqrtΣ = sqrt((L + λ) * Σ)
    χ = [copy(μ) for _ in 1:(2 * L + 1)]
    for l in 2:(L + 1)
        χ[l] .+= sqrtΣ[l - 1, :]
        χ[L + l] .-= sqrtΣ[l - 1, :]
    end
    Y = map(x -> f(model, x), χ)
    μ_y = zeros(T, L)
    Σ_y = zeros(T, L, L)
    for k in 1:(2 * L + 1)
        μ_y .+= Wm[k] .* Y[k]
    end
    for k in 1:(2 * L + 1)
        Σ_y .+= Wc[k] .* (Y[k] - μ_y) * (Y[k] - μ_y)'
    end
    return MvNormalMeanCovariance(μ_y, collect(Hermitian(Σ_y)))
end

@define_message_update_rule(
    node = Flow, target = :out, algorithm = FlowUnscented, args = (m[:in]::MultivariateNormalDistributionsFamily,),
    body = (algo, args) -> flow_unscented_statistics(forward, getmodel(algo), getmethod(algo), mean_cov(args.m[:in])...),
)

@define_message_update_rule(
    node = Flow, target = :in, algorithm = FlowUnscented, args = (m[:out]::MultivariateNormalDistributionsFamily,),
    body = (algo, args) -> flow_unscented_statistics(backward, getmodel(algo), getmethod(algo), mean_cov(args.m[:out])...),
)
