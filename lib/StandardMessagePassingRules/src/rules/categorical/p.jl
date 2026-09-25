# The likelihood p ↦ p_k of an observed category integrates to 1/K! over the simplex of K
# categories, hence the log scale -log K!, in the precision of the probabilities.
categorical_likelihood_logscale(probs) = -convert(float(eltype(probs)), logfactorial(length(probs)))

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
