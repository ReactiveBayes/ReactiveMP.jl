
@node Uniform Stochastic [out, (a, aliases = [α, left]), (b, aliases = [β, right])]

BayesBase.default_prod_rule(::Type{<:Uniform}, ::Type{<:Beta}) = PreserveTypeProd(Distribution)

function Base.prod(::PreserveTypeProd{Distribution}, left::Uniform, right::Beta)
    @assert Distributions.minimum(left) === zero(eltype(left)) && Distributions.maximum(left) === one(eltype(left))
    # The special case for `Uniform(0, 1)` which is essentially `p(x) = 1` and does not change anything
    return right
end

@average_energy Uniform (q_out::Beta, q_a::PointMass, q_b::PointMass) = begin
    @assert (mean(q_a), mean(q_b)) == (0.0, 1.0) "a and b must be equal to 0 and 1 respectively"
    0.0
end
