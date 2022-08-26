
export AbstractFormConstraint
export FormConstraintCheckEach, FormConstraintCheckLast, FormConstraintCheckPickDefault
export constrain_form, default_prod_constraint, default_form_check_strategy, is_point_mass_form_constraint, make_form_constraint
export CompositeFormConstraint

using TupleTools

import Base: +

# Form constraints are preserved during execution of the `prod` function
# There are two major strategies to check current functional form
# We may check and preserve functional form of the result of the `prod` function
# after each subsequent `prod` 
# or we may want to wait after all `prod` functions in the equality chain have been executed 

"""
    AbstractFormConstraint

Every functional form constraint is a subtype of `AbstractFormConstraint` abstract type.

Note: this is not strictly necessary, but it makes automatic dispatch easier and compatible with the `CompositeFormConstraint`.

See also: [`CompositeFormConstraint`](@ref)
"""
abstract type AbstractFormConstraint end

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
    default_prod_constraint(form_constraint)

Returns a default prod constraint needed to apply a given `form_constraint`. For most form constraints this function returns `ProdGeneric`.

See also: [`ProdAnalytical`](@ref), [`ProdGeneric`](@ref)
"""
function default_prod_constraint end

"""
    is_point_mass_form_constraint(form_constraint)

Specifies whether form constraint always returns PointMass estimates or not. For a given `form_constraint` returns either `true` or `false`.

See also: [`FormConstraintCheckEach`](@ref), [`FormConstraintCheckLast`](@ref), [`constrain_form`](@ref)
"""
function is_point_mass_form_constraint end

"""
    constrain_form(form_constraint, distribution)

This function must approximate `distribution` object in a form that satisfies `form_constraint`.

See also: [`FormConstraintCheckEach`](@ref), [`FormConstraintCheckLast`](@ref), [`default_form_check_strategy`](@ref), [`is_point_mass_form_constraint`](@ref)
"""
function constrain_form end

"""
    make_form_constraint(::Type, args...; kwargs...)

Creates form constraint object based on passed `type` with given `args` and `kwargs`. Used to simplify form constraint specification.

As an example:

```julia
make_form_constraint(PointMass)
```

creates an instance of `PointMassFormConstraint` and 

```julia
make_form_constraint(SampleList, 5000, LeftProposal())
```
should create an instance of `SampleListFormConstraint`.

See also: [`AbstractFormConstraint`](@ref)
"""
function make_form_constraint end

"""
    CompositeFormConstraint

Creates a composite form constraint that applies form constraints in order. The composed form constraints must be compatible and have the exact same `form_check_strategy`. 
Any functional form constraint that defines `is_point_mass_form_constraint() = true` may be used only as the last element of the composition.
"""
struct CompositeFormConstraint{C} <: AbstractFormConstraint
    constraints::C
end

Base.show(io::IO, constraint::CompositeFormConstraint) = join(io, constraint.constraints, " :: ")

function constrain_form(composite::CompositeFormConstraint, something)
    return reduce((form, constraint) -> constrain_form(constraint, form), composite.constraints, init = something)
end

function default_prod_constraint(constraint::CompositeFormConstraint)
    return mapfoldl(default_prod_constraint, resolve_prod_constraint, constraint.constraints)
end

function default_form_check_strategy(composite::CompositeFormConstraint)
    strategies = map(default_form_check_strategy, composite.constraints)
    if !(all(e -> e === first(strategies), TupleTools.tail(strategies)))
        error("Different default form check strategy for composite form constraints found. Use `form_check_strategy` options to specify check strategy.")
    end
    return first(strategies)
end

function is_point_mass_form_constraint(composite::CompositeFormConstraint)
    is_point_mass = map(is_point_mass_form_constraint, composite.constraints)
    pmindex       = findnext(is_point_mass, 1)
    if pmindex !== nothing && pmindex !== length(is_point_mass)
        error("Composite form constraint supports point mass constraint only at the end of the form constrains specification.")
    end
    return last(is_point_mass)
end

Base.:+(constraint::AbstractFormConstraint) = constraint

Base.:+(left::AbstractFormConstraint, right::AbstractFormConstraint)   = CompositeFormConstraint((left, right))
Base.:+(left::AbstractFormConstraint, right::CompositeFormConstraint)  = CompositeFormConstraint((left, right.constraints...))
Base.:+(left::CompositeFormConstraint, right::AbstractFormConstraint)  = CompositeFormConstraint((left.constraints..., right))
Base.:+(left::CompositeFormConstraint, right::CompositeFormConstraint) = CompositeFormConstraint((left.constraints..., right.constraints...))

##

struct FormConstraintSpecification{C, P}
    form_constraint::C
    prod_constraint::P
end

const __EmptyFormConstraintSpecification = FormConstraintSpecification(nothing, nothing)

FormConstraintSpecification(form_constraint) = FormConstraintSpecification(form_constraint, default_prod_constraint(form_constraint))

Base.show(io::IO, spec::FormConstraintSpecification) = print(io, spec.form_constraint, " [ prod_constraint = ", spec.prod_constraint, " ]")

function resolve_marginal_messages_form_prod(constraints, model, name)
    marginal_form_constraint, marginal_prod_constraint = resolve_marginal_form_prod(constraints, model, name)
    messages_form_constraint, messages_prod_constraint = resolve_messages_form_prod(constraints, model, name)
    return marginal_form_constraint, messages_form_constraint, resolve_prod_constraint(marginal_prod_constraint, messages_prod_constraint)
end

resolve_marginal_form_prod(constraints, model, name) =
    resolve_form_prod(constraints, model, constraints.marginalsform, name)
resolve_messages_form_prod(constraints, model, name) =
    resolve_form_prod(constraints, model, constraints.messagesform, name)

# Preoptimised dispatch rule for empty form constraints
resolve_form_prod(constraints, model, ::NamedTuple{()}, name) = (nothing, nothing)

function resolve_form_prod(constraints, model, specification, name)
    entry = get(specification, name, __EmptyFormConstraintSpecification)
    return entry.form_constraint, entry.prod_constraint
end
