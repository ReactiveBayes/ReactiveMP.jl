# The message p ↦ ∏ₖ p_k^{q_k} integrates over the simplex to the multivariate beta function
# B(q + 1) = ∏ₖ Γ(q_k + 1) / Γ(K + 1), since the q_k sum to one: its log is the log scale, in the
# precision of the probabilities. For a one-hot q, an observed category, it is -log K!.
function categorical_likelihood_logscale(probs)
    T = float(eltype(probs))
    return sum(q -> loggamma(q + one(T)), probs) - convert(T, logfactorial(length(probs)))
end

@define_message_update_rule(
    node = Categorical, target = :p,
    args = (q[:out]::Categorical,),
    logscale = (args) -> categorical_likelihood_logscale(probvec(args.q[:out])),
    body = (args) -> begin
        probs = probvec(args.q[:out])
        Dirichlet(probs .+ one(eltype(probs)))
    end,
)

@define_message_update_rule(
    node = Categorical, target = :p,
    args = (q[:out]::PointMass{<:AbstractVector{<:Real}},),
    logscale = (args) -> categorical_likelihood_logscale(mean(args.q[:out])),
    body = (args) -> begin
        probs = mean(args.q[:out])
        isonehot(probs) || throw(ArgumentError("q_out must be one-hot encoded. Got: $probs"))
        Dirichlet(probs .+ one(eltype(probs)))
    end,
)

@define_message_update_rule(
    node = Categorical, target = :p,
    args = (q[:out]::Any,),
    body = (args) -> throw(
        ArgumentError(
            "This rule is only defined for PointMass over a one-hot vector or a Categorical distribution. Got: $(typeof(args.q[:out]))",
        ),
    ),
)
