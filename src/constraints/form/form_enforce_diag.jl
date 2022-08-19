export EnforceDiagFormConstraint

import LinearAlgebra: Diagonal

"""
    EnforceDiagFormConstraint

Constrains the marginal of a matrix variate distribution to be diagonal. 
Only applies to (Inverse)Wishart distributions.

# Traits 
- `is_point_mass_form_constraint` = `false`
- `default_form_check_strategy`   = `FormConstraintCheckLast()`
- `default_prod_constraint`       = `ProdAnalytical()`
- `make_form_constraint`          = `EnforceDiagFormConstraint` (for use in `@constraints` macro)

See also: [`constrain_form`](@ref), [`DistProduct`](@ref)
"""

struct EnforceDiagFormConstraint <: AbstractFormConstraint end

ReactiveMP.is_point_mass_form_constraint(::EnforceDiagFormConstraint) = false

ReactiveMP.default_form_check_strategy(::EnforceDiagFormConstraint) = FormConstraintCheckEach()

ReactiveMP.default_prod_constraint(::EnforceDiagFormConstraint) = ProdAnalytical()

ReactiveMP.make_form_constraint(::Type{<:EnforceDiagFormConstraint}) = EnforceDiagFormConstraint()

__drop_offdiag(S::Matrix{Float64}) = Matrix{Float64}(Diagonal(Diagonal(S)))

function ReactiveMP.constrain_form(constraint::EnforceDiagFormConstraint, dist)
    return dist
end

function ReactiveMP.constrain_form(constraint::EnforceDiagFormConstraint, dist::WishartMessage)
    df, S = ReactiveMP.params(dist)
    return WishartMessage(df, __drop_offdiag(S))
end

function ReactiveMP.constrain_form(constraint::EnforceDiagFormConstraint, dist::InverseWishartMessage)
    df, S = ReactiveMP.params(dist)
    return InverseWishartMessage(df, __drop_offdiag(S))
end
