# This file has method implementations for `resolve_prod_constraint`

# In case of both `ProdAnalytical` we simply return the same
resolve_prod_constraint(::ProdAnalytical, ::ProdAnalytical) = ProdAnalytical()

# `ProdPreserveType` overwrites `ProdAnalytical`
resolve_prod_constraint(left::ProdPreserveType, ::ProdAnalytical)  = left
resolve_prod_constraint(::ProdAnalytical, right::ProdPreserveType) = right

resolve_prod_constraint(left::ProdPreserveType{T}, right::ProdPreserveType{T}) where {T}    = ProdPreserveType{T}()
resolve_prod_constraint(left::ProdPreserveType{L}, right::ProdPreserveType{R}) where {L, R} = error("Cannot resolve `prod_constraint`. $(left) and $(right) have different types to preserve.")

resolve_prod_constraint(::ProdPreserveTypeLeft, ::ProdAnalytical)  = ProdPreserveTypeLeft()
resolve_prod_constraint(::ProdPreserveTypeRight, ::ProdAnalytical) = ProdPreserveTypeRight()

resolve_prod_constraint(::ProdAnalytical, ::ProdPreserveTypeLeft)  = ProdPreserveTypeLeft()
resolve_prod_constraint(::ProdAnalytical, ::ProdPreserveTypeRight) = ProdPreserveTypeRight()

resolve_prod_constraint(::ProdPreserveTypeLeft, ::ProdPreserveTypeLeft)   = ProdPreserveTypeLeft()
resolve_prod_constraint(::ProdPreserveTypeRight, ::ProdPreserveTypeRight) = ProdPreserveTypeRight()

resolve_prod_constraint(::ProdPreserveTypeLeft, ::ProdPreserveTypeRight) = error("Cannot resolve `prod_constraint`. `ProdPreserveTypeLeft()` and `ProdPreserveTypeRight()` are incompatible.")
resolve_prod_constraint(::ProdPreserveTypeRight, ::ProdPreserveTypeLeft) = error("Cannot resolve `prod_constraint`. `ProdPreserveTypeRight()` and `ProdPreserveTypeLeft()` are incompatible.")

# `ProdGeneric` always has "inner" prod_constraint, e.g. `ProdAnalytical` to resolve analytical cases, but it might be any other prod_constraint as well
resolve_prod_constraint(left::ProdGeneric, right::AbstractProdConstraint) = ProdGeneric(resolve_prod_constraint(get_constraint(left), right))
resolve_prod_constraint(left::AbstractProdConstraint, right::ProdGeneric) = ProdGeneric(resolve_prod_constraint(left, get_constraint(right)))

resolve_prod_constraint(left::ProdGeneric, right::ProdGeneric) = ProdGeneric(resolve_prod_constraint(get_constraint(left), get_constraint(right)))

# `ProdFinal` has the higher priority among the others
resolve_prod_constraint(left::ProdFinal, right::AbstractProdConstraint) = left
resolve_prod_constraint(left::AbstractProdConstraint, right::ProdFinal) = right

resolve_prod_constraint(left::ProdFinal, right::ProdFinal) = error("Cannot resolve `prod_constraint`. Both $(left) and $(right) prod constraints are of type `ProdFinal`")

# Utility case
resolve_prod_constraint(left::Nothing, right::AbstractProdConstraint) = right
resolve_prod_constraint(left::AbstractProdConstraint, right::Nothing) = left
resolve_prod_constraint(::Nothing, ::Nothing) = nothing
