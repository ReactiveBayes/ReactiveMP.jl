# Flow's rules: a normal pushed through the model, forwards towards `out` and backwards towards
# `in`. Linearization uses the model's own Jacobians; Unscented, sigma points.

const FlowLinearization = FlowApproximation{<:AbstractCompiledFlowModel, <:Linearization}
const FlowUnscented = FlowApproximation{<:AbstractCompiledFlowModel, <:Unscented}

@define_message_update_rule(
    node = Flow, target = :out, algorithm = FlowLinearization, args = (m[:in]::MvNormalMeanCovariance,),
    body = (algo, args) -> begin
        μ_in, Σ_in = mean_cov(args.m[:in])
        μ_out, J = forward_jacobian(getmodel(algo), μ_in)
        MvNormalMeanCovariance(μ_out, J * Σ_in * J')
    end,
)

# In precision form the inverse Jacobian carries the precision, for a mean-precision or a
# weighted-mean-precision input alike.
flow_forward_precision(algo, m_in) = begin
    μ_in, Λ_in = mean_precision(m_in)
    Ji = inv_jacobian(getmodel(algo), μ_in)
    MvNormalMeanPrecision(forward(getmodel(algo), μ_in), Ji' * Λ_in * Ji)
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

flow_backward_precision(algo, m_out) = begin
    μ_out, Λ_out = mean_precision(m_out)
    J = jacobian(getmodel(algo), μ_out)
    MvNormalMeanPrecision(backward(getmodel(algo), μ_out), J' * Λ_out * J)
end

@define_message_update_rule(
    node = Flow, target = :in, algorithm = FlowLinearization, args = (m[:out]::MvNormalMeanPrecision,),
    body = (algo, args) -> flow_backward_precision(algo, args.m[:out]),
)

@define_message_update_rule(
    node = Flow, target = :in, algorithm = FlowLinearization, args = (m[:out]::MvNormalWeightedMeanPrecision,),
    body = (algo, args) -> flow_backward_precision(algo, args.m[:out]),
)

# The unscented method with its weights for dimension `dim`: `Unscented(dim)` as given, or built
# from `Unscented()`'s parameters.
function flow_unscented(method::Unscented, dim)
    if method.e === nothing
        return Unscented(dim; alpha = method.α, beta = method.β, kappa = method.κ)
    end
    L = MessagePassingRulesApproximations.getL(method)
    L == dim || throw(DimensionMismatch("`Unscented($L)` was given for a flow of dimension $dim"))
    return method
end

# The normal of `f(x)` for `x ~ N(μ, Σ)`, by v6's sigma points: the rows of the symmetric square
# root of `(L + λ) Σ`. The numerics package's `unscented_statistics` takes a Cholesky factor,
# which gives a nonlinear flow other results, so v6's are kept.
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
