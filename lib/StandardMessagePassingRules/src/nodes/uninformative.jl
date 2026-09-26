"""
    Uninformative()

The uninformative node, a factor of one over its single interface `out`, and the value of its
message. The rule towards `out` takes no inputs and returns `Uninformative()` itself; it declares
no log scale, so the message's log scale is undefined. `Uninformative()` is the identity of `prod` under the default product
rule, so it changes nothing it is multiplied with. The average energy is zero.

A model uses it to leave a variable without a prior, `x ~ Uninformative()`, where every other
factor on `x` then decides its marginal. `Uninformative()` is not a distribution and has no
mean or density, so a variable needs some other factor for its marginal to be one.

# Examples

```jldoctest; setup = :(using StandardMessagePassingRules, BayesBase, ExponentialFamily)
julia> prior = NormalMeanVariance(0.0, 1.0);

julia> prod(GenericProd(), Uninformative(), prior) === prior
true

julia> prod(GenericProd(), Uninformative(), Uninformative())
Uninformative()
```

See also [`StandaloneDistribution`](@ref).
"""
struct Uninformative end

@define_factor_node(node = Uninformative, type = Stochastic, interfaces = [:out])

@define_average_energy(
    node = Uninformative,
    args = (q[:out]::Any,),
    body = (args) -> zero(paramfloattype(args.q[:out])),
)

"""
    UninformativeProd()

The product strategy for a product with [`Uninformative`](@ref): `prod(UninformativeProd(), u, x)`
is `x`, `prod(UninformativeProd(), x, u)` is `x`, and two `Uninformative()` give
`Uninformative()`. `BayesBase.default_prod_rule` selects it whenever either side is
`Uninformative`, so `prod(GenericProd(), …)` and the engine's products use it.

The strategy is this package's own, so its `prod` methods cannot be ambiguous with anyone
else's, as generic `PreserveTypeProd{T}` methods would be. Five of its `default_prod_rule`
methods settle the overlaps with BayesBase's own rules for lazy products (`LinearizedProductOf`),
mixtures (`MixtureDistribution`) and terminal arguments (`TerminalProdArgument`). They encode
BayesBase's rules, so a new rule there can bring an ambiguity back; the lasting fix is for
BayesBase to own `Uninformative` as a product identity, as it treats `missing`.
"""
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
