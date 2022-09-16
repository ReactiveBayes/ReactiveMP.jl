
@node Uniform Stochastic [out, (a, aliases = [α, left]), (b, aliases = [β, right])]

prod_analytical_rule(::Type{<:Uniform}, ::Type{<:Beta}) = ProdAnalyticalRuleAvailable()
prod_analytical_rule(::Type{<:Beta}, ::Type{<:Uniform}) = ProdAnalyticalRuleAvailable()

prod(::ProdAnalytical, left::Beta, right::Uniform) = prod(ProdAnalytical(), right, left)

function prod(::ProdAnalytical, left::Uniform, right::Beta)
    @assert Distributions.minimum(left) === zero(eltype(left)) && Distributions.maximum(left) === one(eltype(left))
    # The special case for `Uniform(0, 1)` which is essentially `p(x) = 1` and does not change anything
    return right
end

@average_energy Uniform (q_out::Any, q_a::PointMass, q_b::PointMass) = begin
    log(mean(q_b)-mean(q_a))
end