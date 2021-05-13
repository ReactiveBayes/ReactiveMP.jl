export ProdAnalytical

import Base: prod

"""
    ProdAnalytical

`ProdAnalytical` is one of the strategies for `prod` function. This strategy uses analytical prod methods but does not constraint a prod to be in any specific form.
It fails if no analytical rules is available, use `ProdGeneric` prod strategy to fallback to approximation methods.

See also: [`prod`](@ref), [`ProdPreserveType`](@ref), [`ProdGeneric`](@ref)
"""
struct ProdAnalytical end

prod(::ProdAnalytical, left, right) = error("No analytical rule available to compute a product of distributions $(left) and $(right).")