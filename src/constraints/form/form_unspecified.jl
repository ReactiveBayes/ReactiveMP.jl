export UnspecifiedFormConstraint

"""
    UnspecifiedFormConstraint

One of the form constraint objects. Does not imply any form constraints and simply returns the same object as receives.
However it does not allow `DistProduct` to be a valid functional form in the inference backend.

See also: [`constrain_form`](@ref), [`DistProduct`](@ref)
"""
struct UnspecifiedFormConstraint <: AbstractFormConstraint end

is_point_mass_form_constraint(::UnspecifiedFormConstraint) = false

default_form_check_strategy(::UnspecifiedFormConstraint) = FormConstraintCheckLast()

default_prod_constraint(::UnspecifiedFormConstraint) = ProdAnalytical()

make_form_constraint(::Type{ <: Nothing }) = UnspecifiedFormConstraint()

constrain_form(::UnspecifiedFormConstraint, something)                            = something
constrain_form(::UnspecifiedFormConstraint, something::Message{ <: DistProduct }) = error("`DistProduct` object cannot be used as a functional form in inference backend. Use form constraints to restrict the functional form of marginal posteriors.")