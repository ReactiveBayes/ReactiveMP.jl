export SampleListFormConstraint

"""
    SampleListFormConstraint

One of the form constraint objects. Approximates `DistProduct` with a SampleList object. 

See also: [`constrain_form`](@ref), [`DistProduct`](@ref)
"""
struct SampleListFormConstraint <: AbstractFormConstraint end

default_form_check_strategy(::SampleListFormConstraint) = FormConstraintCheckLast()

is_point_mass_form_constraint(::SampleListFormConstraint) = false

constrain_form(::SampleListFormConstraint, something) = something

function constrain_form(constraint::SampleListFormConstraint, something::Message{ <: DistProduct })
    product       = getdata(something)
    left          = constrain_form(constraint, getleft(product))
    right         = constrain_form(constraint, getright(product))
    # TODO: Check the best strategy of passing arguments here
    # For now its (right, left), since left is considered to be a prior and right is almost always a likelihood
    approximation = approximate_prod_with_sample_list(right, left) # right, left is intentional
    return Message(approximation, is_clamped(something), is_initial(something))
end