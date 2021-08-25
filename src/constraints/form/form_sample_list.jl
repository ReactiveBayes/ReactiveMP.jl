export UnspecifiedFormConstraint

"""
    SampleListFormConstraint

One of the form constraint objects. Approximates `DistProduct` with a SampleList object. 

See also: [`constrain_form`](@ref), [`DistProduct`](@ref)
"""
struct SampleListFormConstraint <: AbstractFormConstraint end

default_form_check_strategy(::UnspecifiedFormConstraint) = FormConstraintCheckLast()

is_point_mass_form_constraint(::UnspecifiedFormConstraint) = false

constrain_form(::SampleListFormConstraint, something) = something

function constrain_form(constraint::SampleListFormConstraint, something::Message{ <: DistProduct })
    product       = getdata(something)
    left          = constrain_form(constraint, getleft(product))
    right         = constrain_form(constraint, getright(product))
    approximation = approximate_prod_with_sample_list(left, right)
    return Message(approximation, is_clamped(something), is_initial(something))
end