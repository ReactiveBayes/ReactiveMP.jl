"""
    Uninformative

The uninformative node: a factor of 1 over `out`, whose message changes nothing it is
multiplied with.
"""
struct Uninformative end

@define_factor_node(node = Uninformative, type = Stochastic, interfaces = [:out])

@define_average_energy(
    node = Uninformative,
    args = (q[:out]::Any,),
    body = (args) -> zero(paramfloattype(args.q[:out])),
)

# The product with an `Uninformative` is the other factor. The strategy is this package's own,
# so its `prod` methods cannot be ambiguous with anyone else's, as generic
# `PreserveTypeProd{T}` methods would be. The last five `default_prod_rule`
# methods settle the overlaps with BayesBase's own rules for lazy products, mixtures and
# terminal arguments. They encode BayesBase's rules, so a new one there can bring an ambiguity
# back; the fix is for BayesBase to own `Uninformative` as a product identity, as it treats
# `missing`.
struct UninformativeProd end

BayesBase.default_prod_rule(::Type{<:Uninformative}, ::Type) = UninformativeProd()
BayesBase.default_prod_rule(::Type, ::Type{<:Uninformative}) = UninformativeProd()
BayesBase.default_prod_rule(::Type{<:Uninformative}, ::Type{<:Uninformative}) = UninformativeProd()
BayesBase.default_prod_rule(::Type{LinearizedProductOf{F}}, ::Type{F}) where {F <: Uninformative} = UninformativeProd()
BayesBase.default_prod_rule(::Type{F}, ::Type{LinearizedProductOf{F}}) where {F <: Uninformative} = UninformativeProd()
BayesBase.default_prod_rule(::Type{<:MixtureDistribution}, ::Type{<:Uninformative}) = UninformativeProd()
BayesBase.default_prod_rule(::Type{<:Uninformative}, ::Type{<:TerminalProdArgument}) = UninformativeProd()
BayesBase.default_prod_rule(::Type{<:TerminalProdArgument}, ::Type{<:Uninformative}) = UninformativeProd()

BayesBase.prod(::UninformativeProd, ::Uninformative, right) = right
BayesBase.prod(::UninformativeProd, left, ::Uninformative) = left
BayesBase.prod(::UninformativeProd, ::Uninformative, ::Uninformative) = Uninformative()
