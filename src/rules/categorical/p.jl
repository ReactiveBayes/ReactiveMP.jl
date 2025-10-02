export rule

# Check if a vector is one-hot
_isonehot(vec::AbstractVector) = sum(vec .== 1.0) == 1 && all(x -> x == 0.0 || x == 1.0, vec)

function _validated_dirichlet(q_out; logscale::Bool=false)
    probs = probvec(q_out)
    if !_isonehot(probs) throw(ArgumentError("q_out must be one-hot encoded. Got: $probs")) end
    if logscale @logscale -logfactorial(length(probs)) end
    return Dirichlet(probs .+ one(eltype(probs)))
end

using SpecialFunctions: logfactorial
# https://github.com/JuliaStats/Distributions.jl/blob/master/src/univariate/discrete/categorical.jl#L27
@rule Categorical(:p, Marginalisation) (q_out::Categorical{P,Ps},) where {P<:Real, Ps<:AbstractVector{P}} = _validated_dirichlet(q_out)
# https://github.com/ReactiveBayes/BayesBase.jl/blob/main/src/densities/pointmass.jl
@rule Categorical(:p, Marginalisation) (q_out::PointMass{V},) where {T<:Real, V<:AbstractVector{T}} = begin
        probs = probvec(q_out)
    if !_isonehot(probs) throw(ArgumentError("q_out must be one-hot encoded. Got: $probs")) end
    @logscale -logfactorial(length(probs))
    return Dirichlet(probs .+ one(eltype(probs)))
end
@rule Categorical(:p, Marginalisation) (q_out::Any,) = throw(
    ArgumentError("q_out is only defined for PointMass or Categorical over a one-hot vector. Got: $(typeof(q_out))")
)
