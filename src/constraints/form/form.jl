export FormConstraintCheckEach, FormConstraintCheckLast, FormConstraintCheckPickDefault
export constrain_form, default_form_check_strategy, is_point_mass_form_constraint

# Form constraints are preserved during execution of the `prod` function
# There are two major strategies to check current functional form
# We may check and preserve functional form of the result of the `prod` function
# after each subsequent `prod` 
# or we may want to wait after all `prod` functions in the equality chain have been executed 

"""
    FormConstraintCheckEach

This form constraint check strategy checks functional form of the messages product after each product in an equality chain. 
Usually if a variable has been connected to multiple nodes we want to perform multiple `prod` to obtain a posterior marginal.
With this form check strategy `constrain_form` function will be executed after each subsequent `prod` function.

See also: [`FormConstraintCheckLast`](@ref), [`default_form_check_strategy`](@ref), [`constrain_form`](@ref), [`multiply_messages`](@ref)
"""
struct FormConstraintCheckEach end

"""
    FormConstraintCheckEach

This form constraint check strategy checks functional form of the last messages product in the equality chain. 
Usually if a variable has been connected to multiple nodes we want to perform multiple `prod` to obtain a posterior marginal.
With this form check strategy `constrain_form` function will be executed only once after all subsequenct `prod` functions have been executed.

See also: [`FormConstraintCheckLast`](@ref), [`default_form_check_strategy`](@ref), [`constrain_form`](@ref), [`multiply_messages`](@ref)
"""
struct FormConstraintCheckLast end

"""
    FormConstraintCheckPickDefault

This form constraint check strategy simply fallbacks to a default check strategy for a given form constraint. 

See also: [`FormConstraintCheckEach`](@ref), [`FormConstraintCheckLast`](@ref), [`default_form_check_strategy`](@ref)
"""
struct FormConstraintCheckPickDefault end

"""
    default_form_check_strategy(form_constraint)

Returns a default check strategy (e.g. `FormConstraintCheckEach` or `FormConstraintCheckEach`) for a given form constraint object.

See also: [`FormConstraintCheckEach`](@ref), [`FormConstraintCheckLast`](@ref), [`constrain_form`](@ref)
"""
function default_form_check_strategy end

"""
    is_point_mass_form_constraint(form_constraint)

Specifies whether form constraint always returns PointMass estimates or not. For a given `form_constraint` returns either `true` or `false`.

See also: [`FormConstraintCheckEach`](@ref), [`FormConstraintCheckLast`](@ref), [`constrain_form`](@ref)
"""
function is_point_mass_form_constraint end

"""
    constrain_form(form_constraint, something)

Checks that the functional form of `something` object fits `form_constraint` constraint.
If functional form of `something` object does not fit the given constraint this function 
should try to approximate `something` object to be in line with the given `form_constraint`.

See also: [`FormConstraintCheckEach`](@ref), [`FormConstraintCheckLast`](@ref), [`default_form_check_strategy`](@ref), [`is_point_mass_form_constraint`](@ref)
"""
function constrain_form end 


