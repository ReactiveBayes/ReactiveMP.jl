export ProdNoConstraints

import Base: prod

"""
    ProdNoConstraints

`ProdNoConstraints` is one of the strategies for `prod` function. This strategy does not constraint a prod to be in any specific form.
In most of the cases it uses analytical rule to compute a product of two distributions and fails otherwise.

See also: [`prod`](@ref), [`ProdPreserveType`](@ref), [`ProdPointMass`](@ref)
"""
struct ProdNoConstraints end

prod(::ProdNoConstraints, left, right) = error("No analytical rule available to compute a product of distributions $(left) and $(right).")