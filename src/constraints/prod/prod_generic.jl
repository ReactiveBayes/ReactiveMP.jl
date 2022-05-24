export DistProduct, ProdGeneric

import Distributions
import Base: prod, show

"""
    DistProduct

If inference backend cannot return an analytical solution for a product of two distributions it may fallback to the `DistProduct` structure
`DistProduct` is useful to propagate the exact forms of two messages until it hits some approximation method for form-constraint.
However `DistProduct` cannot be used to compute statistics such as mean or variance. 
It has to be approximated before using in actual inference procedure.

Backend exploits form constraints specification which usually help to deal with intractable distributions products. 

See also: [`prod`](@ref), [`ProdGeneric`](@ref)
"""
struct DistProduct{L, R}
    left  :: L
    right :: R
end

Base.show(io::IO, product::DistProduct) = print(io, "DistProduct(", getleft(product), ",", getright(product), ")")

getleft(product::DistProduct)  = product.left
getright(product::DistProduct) = product.right

function Distributions.support(product::DistProduct)
    lsupport = Distributions.support(getleft(product))
    rsupport = Distributions.support(getright(product))
    if lsupport != rsupport
        error("Product $product has different support for left and right entries.")
    end
    return lsupport
end

Distributions.mean(product::DistProduct)      = error("mean() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.median(product::DistProduct)    = error("median() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.mode(product::DistProduct)      = error("mode() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.shape(product::DistProduct)     = error("shape() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.scale(product::DistProduct)     = error("scale() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.rate(product::DistProduct)      = error("rate() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.var(product::DistProduct)       = error("var() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.std(product::DistProduct)       = error("std() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.cov(product::DistProduct)       = error("cov() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.invcov(product::DistProduct)    = error("invcov() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.logdetcov(product::DistProduct) = error("logdetcov() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.entropy(product::DistProduct)   = error("entropy() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.params(product::DistProduct)    = error("params() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")

Distributions.pdf(product::DistProduct, x)    = Distributions.pdf(product.left, x) * Distributions.pdf(product.right, x)
Distributions.logpdf(product::DistProduct, x) = Distributions.logpdf(product.left, x) + Distributions.logpdf(product.right, x)

Base.precision(product::DistProduct) = error("precision() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
Base.length(product::DistProduct)    = error("length() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
Base.ndims(product::DistProduct)     = error("ndims() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
Base.size(product::DistProduct)      = error("size() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")

probvec(product::DistProduct)      = error("probvec() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
weightedmean(product::DistProduct) = error("weightedmean() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")

variate_form(::P) where {P <: DistProduct}           = variate_form(P)
variate_form(::Type{DistProduct{L, R}}) where {L, R} = _check_dist_product_variate_form(variate_form(L), variate_form(R))

_check_dist_product_variate_form(::Type{F}, ::Type{F}) where {F <: VariateForm}                       = F
_check_dist_product_variate_form(::Type{F1}, ::Type{F2}) where {F1 <: VariateForm, F2 <: VariateForm} = error("DistProduct has different variate forms for left ($F1) and right ($F2) entries.")

value_support(::P) where {P <: DistProduct}           = value_support(P)
value_support(::Type{DistProduct{L, R}}) where {L, R} = _check_dist_product_value_support(value_support(L), value_support(R))

_check_dist_product_value_support(::Type{S}, ::Type{S}) where {S <: ValueSupport}                        = S
_check_dist_product_value_support(::Type{S1}, ::Type{S2}) where {S1 <: ValueSupport, S2 <: ValueSupport} = error("DistProduct has different value supports for left ($S1) and right ($S2) entries.")

"""
    ProdGeneric{C}

`ProdGeneric` is one of the strategies for `prod` function. This strategy does not fail in case of no analytical rule is available, but simply creates a product tree, there all nodes represent the `prod` function and all leaves are valid `Distribution` object.
This object does not define any statistical properties (such as `mean` or `var` etc) and cannot be used during the inference procedure. However this object plays imporant part in the functional form constraints implementation. 
In a few words this object keeps all the information of a product of messages and propagates this information in the functional form constraint.

`ProdGeneric` has a "fallback" method, which it may or may not use under some circumstances. For example if the `fallback` method is `ProdAnalytical` (which is the default one) - `ProdGeneric` will try to optimize `prod` tree with analytical solutions where possible.

See also: [`prod`](@ref), [`DistProduct`](@ref), [`ProdAnalytical`](@ref), [`ProdPreserveType`](@ref), [`prod_analytical_rule`](@ref)
"""
struct ProdGeneric{C} <: AbstractProdConstraint
    prod_constraint::C
end

Base.show(io::IO, prod::ProdGeneric) = print(io, "ProdGeneric(fallback = ", prod.prod_constraint, ")")

get_constraint(prod_generic::ProdGeneric) = prod_generic.prod_constraint

ProdGeneric() = ProdGeneric(ProdAnalytical())

prod(::ProdGeneric, ::Missing, right)     = right
prod(::ProdGeneric, left, ::Missing)      = left
prod(::ProdGeneric, ::Missing, ::Missing) = missing

prod(generic::ProdGeneric, left::L, right::R) where {L, R} = prod(generic, prod_analytical_rule(L, R), left, right)

prod(generic::ProdGeneric, ::ProdAnalyticalRuleAvailable, left, right) = prod(get_constraint(generic), left, right)
prod(generic::ProdGeneric, ::ProdAnalyticalRuleUnknown, left, right)   = DistProduct(left, right)

# In case of ProdPointMass we want to propagate a single `DistProduct` as much as possible and do not create a big tree of product which will reduce performance significantly
# In this methods the general rule is the folowing: If we see that one of the arguments of `DistProduct` has the same function form 
# as second argument of `prod` function it is better to try to `prod` them together with `NoConstraint` strategy.
prod(generic::ProdGeneric, left::DistProduct{L, R}, right::T) where {L, R, T} =
    prod(generic, prod_analytical_rule(L, T), prod_analytical_rule(R, T), left, right)
prod(generic::ProdGeneric, left::T, right::DistProduct{L, R}) where {L, R, T} =
    prod(generic, prod_analytical_rule(T, L), prod_analytical_rule(T, R), left, right)

prod(generic::ProdGeneric, ::ProdAnalyticalRuleUnknown, ::ProdAnalyticalRuleUnknown, left::DistProduct, right)   = DistProduct(left, right)
prod(generic::ProdGeneric, ::ProdAnalyticalRuleAvailable, ::ProdAnalyticalRuleUnknown, left::DistProduct, right) = DistProduct(prod(get_constraint(generic), getleft(left), right), getright(left))
prod(generic::ProdGeneric, ::ProdAnalyticalRuleUnknown, ::ProdAnalyticalRuleAvailable, left::DistProduct, right) = DistProduct(getleft(left), prod(get_constraint(generic), getright(left), right))

prod(generic::ProdGeneric, ::ProdAnalyticalRuleUnknown, ::ProdAnalyticalRuleUnknown, left, right::DistProduct)   = DistProduct(left, right)
prod(generic::ProdGeneric, ::ProdAnalyticalRuleAvailable, ::ProdAnalyticalRuleUnknown, left, right::DistProduct) = DistProduct(prod(get_constraint(generic), left, getleft(right)), getright(right))
prod(generic::ProdGeneric, ::ProdAnalyticalRuleUnknown, ::ProdAnalyticalRuleAvailable, left, right::DistProduct) = DistProduct(getleft(right), prod(get_constraint(generic), left, getright(right)))

function prod(generic::ProdGeneric, left::DistProduct{L1, R1}, right::DistProduct{L2, R2}) where {L1, R1, L2, R2}
    return prod(
        generic,
        prod_analytical_rule(L1, L2), prod_analytical_rule(L1, R2),
        prod_analytical_rule(R1, L2), prod_analytical_rule(R1, R2),
        left, right
    )
end

prod(::ProdGeneric, _, _, _, _, left::DistProduct, right::DistProduct) = DistProduct(left, right)

function prod(
    generic::ProdGeneric,
    ::ProdAnalyticalRuleAvailable,
    _,
    _,
    ::ProdAnalyticalRuleAvailable,
    left::DistProduct,
    right::DistProduct
)
    return prod(
        generic,
        prod(get_constraint(generic), getleft(left), getleft(right)),
        prod(get_constraint(generic), getright(left), getright(right))
    )
end
