@define_message_update_rule(
    node = Categorical, target = :p,
    args = (q[:out]::Categorical,),
    body = (args, ann) -> begin
        probs = probvec(args.q[:out])
        annotate!(ann, :logscale, -logfactorial(length(probs)))
        Dirichlet(probs .+ one(eltype(probs)))
    end,
)

@define_message_update_rule(
    node = Categorical, target = :p,
    args = (q[:out]::PointMass{<:AbstractVector{<:Real}},),
    body = (args, ann) -> begin
        probs = mean(args.q[:out])
        isonehot(probs) || throw(ArgumentError("q_out must be one-hot encoded. Got: $probs"))
        annotate!(ann, :logscale, -logfactorial(length(probs)))
        Dirichlet(probs .+ one(eltype(probs)))
    end,
)
