export ProdAnalytical

import Base: prod, showerror

"""
    ProdAnalytical

`ProdAnalytical` is one of the strategies for `prod` function. This strategy uses analytical prod methods but does not constraint a prod to be in any specific form.
It throws an `NoAnalyticalProdException` if no analytical rules is available, use `ProdGeneric` prod strategy to fallback to approximation methods.

Note: `ProdAnalytical` ignores `missing` values and simply returns the non-`missing` argument. Returns `missing` in case if both arguments are `missing`.

See also: [`prod`](@ref), [`ProdPreserveType`](@ref), [`ProdGeneric`](@ref)
"""
struct ProdAnalytical <: AbstractProdConstraint end

"""
    prod(strategy, left, right)

`prod` function is used to find a product of two probability distrubutions (or any other objects) over same variable (e.g. 𝓝(x|μ_1, σ_1) × 𝓝(x|μ_2, σ_2)).
There are multiple strategies for prod function, e.g. `ProdAnalytical`, `ProdGeneric` or `ProdPreserveType`.

# Examples:
```jldoctest
using ReactiveMP

product = prod(ProdAnalytical(), NormalMeanVariance(-1.0, 1.0), NormalMeanVariance(1.0, 1.0))

mean(product), var(product)

# output
(0.0, 0.5)
```

See also: [`prod_analytical_rule`](@ref), [`ProdAnalytical`](@ref), [`ProdGeneric`](@ref)
"""
prod(::ProdAnalytical, left, right) = throw(NoAnalyticalProdException(left, right))

prod(::ProdAnalytical, ::Missing, right)     = right
prod(::ProdAnalytical, left, ::Missing)      = left
prod(::ProdAnalytical, ::Missing, ::Missing) = missing

"""
    NoAnalyticalProdException(left, right)

This exception is thrown in the `prod` function in case if an analytical prod between `left` and `right` is not available or not implemented.

See also: [`ProdAnalytical`](@ref), [`prod`]
"""
struct NoAnalyticalProdException{L, R} <: Exception
    left  :: L 
    right :: R
end

function Base.showerror(io::IO, err::NoAnalyticalProdException)
    print(io, "NoAnalyticalProdException: ")
    print(io, "  No analytical rule available to compute a product of $(err.left) and $(err.right).")
    print(io, "  Possible fix: implement `prod(::ProdAnalytical, left::$(typeof(err.left)), right::$(typeof(err.right))) = ...`")
end

