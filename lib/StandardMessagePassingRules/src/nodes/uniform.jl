@define_factor_node(node = Uniform, type = Stochastic, interfaces = [:out, (:a, aliases = [:α, :left]), (:b, aliases = [:β, :right])])

# Uniform(0, 1) times a Beta is the Beta, a special case which neither package that owns
# these types defines. It is piracy, declared as owned in the quality tests, and a candidate
# for ExponentialFamily.
BayesBase.default_prod_rule(::Type{<:Uniform}, ::Type{<:Beta}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::Uniform, right::Beta)
    (minimum(left) == zero(eltype(left)) && maximum(left) == one(eltype(left))) ||
        throw(ArgumentError("the product of a Uniform and a Beta is defined only for Uniform(0, 1), got $left"))
    return right
end

@define_average_energy(
    node = Uniform,
    args = (q[:out]::Beta, q[:a]::PointMass, q[:b]::PointMass),
    body = (args) -> begin
        (mean(args.q[:a]), mean(args.q[:b])) == (0, 1) || throw(ArgumentError("a and b must be equal to 0 and 1 respectively"))
        zero(paramfloattype(args.q[:out]))
    end,
)
