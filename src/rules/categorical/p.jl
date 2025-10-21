
using SpecialFunctions: logfactorial

# https://github.com/JuliaStats/Distributions.jl/blob/master/src/univariate/discrete/categorical.jl#L27
@rule Categorical(:p, Marginalisation) (q_out::Categorical,) = begin
    probs = probvec(q_out)
    @logscale -logfactorial(length(probs))
    return Dirichlet(probs .+ one(eltype(probs)))
end

# https://github.com/ReactiveBayes/BayesBase.jl/blob/main/src/densities/pointmass.jl
@rule Categorical(:p, Marginalisation) (q_out::PointMass{V},) where {T <: Real, V <: AbstractVector{T}} = begin
    probs = mean(q_out)
    if !isonehot(probs)
        throw(ArgumentError("q_out must be one-hot encoded. Got: $probs"))
    end
    @logscale -logfactorial(length(probs))
    return Dirichlet(probs .+ one(eltype(probs)))
end

@rule Categorical(:p, Marginalisation) (q_out::Any,) = throw(
    ArgumentError("This rule is only defined for PointMass over a one-hot vector or a Categorical distribution. Got: $(typeof(q_out))")
)
