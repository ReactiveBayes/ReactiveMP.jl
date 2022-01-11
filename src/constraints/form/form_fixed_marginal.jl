export FixedMarginalConstraint

"""
FixedMarginalConstraint

One of the form constraint objects. 
Provides a constraint on the marginal distribution such that it remains fixed during inference. 
Can be viewed as blocking of updates of a specific edge associated with the marginal. 

See also: [`constrain_form`](@ref), [`DistProduct`](@ref)
"""

mutable struct FixedMarginalConstraint <: ReactiveMP.AbstractFormConstraint
    fixed_value :: Any
end

default_form_check_strategy(::FixedMarginalConstraint) = FormConstraintCheckLast()

is_point_mass_form_constraint(::FixedMarginalConstraint) = false

function constrain_form(constraint::FixedMarginalConstraint, something)
    if constraint.fixed_value !== nothing
        return Message(constraint.fixed_value, false, false)
    else 
        return something
    end
end 