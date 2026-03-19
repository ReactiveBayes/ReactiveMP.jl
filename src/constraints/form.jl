export AbstractFormConstraint
export FormConstraintCheckEach,
    FormConstraintCheckLast, FormConstraintCheckPickDefault
export constrain_form, default_prod_constraint, default_form_check_strategy
export UnspecifiedFormConstraint, CompositeFormConstraint

using TupleTools

import BayesBase: resolve_prod_strategy
import Base: +

# Form constraints control the functional form of messages during the product computation.
# There are two strategies for when to apply the constraint:
# - `FormConstraintCheckEach`: apply after each pairwise `prod` in `compute_product_of_two_messages`
# - `FormConstraintCheckLast`: apply once at the end in `compute_product_of_messages`

"""
    AbstractFormConstraint

Abstract supertype for all form constraints. Subtype this to create custom form constraints
that can be used with [`constrain_form`](@ref) and [`ReactiveMP.MessageProductContext`](@ref).

Not strictly required (any object works via [`ReactiveMP.WrappedFormConstraint`](@ref)),
but makes dispatch easier and is needed for [`ReactiveMP.CompositeFormConstraint`](@ref) composition via `+`.
"""
abstract type AbstractFormConstraint end

"""
    FormConstraintCheckEach

Form constraint check strategy that applies [`constrain_form`](@ref) after **each** pairwise product
inside [`ReactiveMP.compute_product_of_two_messages`](@ref). Use this when intermediate results
need to stay in a specific functional form (e.g. to prevent numerical issues during long product chains).

See also: [`FormConstraintCheckLast`](@ref), [`ReactiveMP.MessageProductContext`](@ref)
"""
struct FormConstraintCheckEach end

"""
    FormConstraintCheckLast

Form constraint check strategy that applies [`constrain_form`](@ref) **once** at the very end
of [`ReactiveMP.compute_product_of_messages`](@ref), after all pairwise products have been folded.
This is the default strategy and is more efficient when intermediate form doesn't matter.

See also: [`FormConstraintCheckEach`](@ref), [`ReactiveMP.MessageProductContext`](@ref)
"""
struct FormConstraintCheckLast end

"""
    FormConstraintCheckPickDefault

A meta-strategy that defers to the default check strategy of the given form constraint,
as defined by [`default_form_check_strategy`](@ref).
"""
struct FormConstraintCheckPickDefault end

"""
    default_form_check_strategy(form_constraint)

Returns the default check strategy (either [`FormConstraintCheckEach`](@ref) or [`FormConstraintCheckLast`](@ref))
for a given form constraint. Override this for custom constraints to control when they are applied.
"""
function default_form_check_strategy end

"""
    default_prod_constraint(form_constraint)

Returns the default product strategy needed to apply a given `form_constraint`.
For most form constraints this returns `BayesBase.GenericProd()`.
"""
function default_prod_constraint end

"""
    constrain_form(constraint, distribution)

Applies the form `constraint` to `distribution` and returns the constrained result.
This is the main extension point for custom form constraints — implement a method of this function
for your constraint type and the distribution types you want to support.

See also: [`AbstractFormConstraint`](@ref), [`ReactiveMP.MessageProductContext`](@ref)
"""
function constrain_form end

"""
    UnspecifiedFormConstraint

The default form constraint — does nothing and returns the distribution as-is.
Used when no form constraint has been specified in the [`ReactiveMP.MessageProductContext`](@ref).
"""
struct UnspecifiedFormConstraint <: AbstractFormConstraint end

default_form_check_strategy(::UnspecifiedFormConstraint) = FormConstraintCheckLast()

default_prod_constraint(::UnspecifiedFormConstraint) = GenericProd()

constrain_form(::UnspecifiedFormConstraint, something) = something

"""
    WrappedFormConstraint(constraint, context)

A wrapper that pairs a form constraint with an optional precomputed context.
Any object that is not a subtype of [`AbstractFormConstraint`](@ref) gets automatically wrapped
into this during [`ReactiveMP.preprocess_form_constraints`](@ref).
Use [`ReactiveMP.prepare_context`](@ref) to provide extra context that can be reused across multiple [`constrain_form`](@ref) calls.
"""
struct WrappedFormConstraint{C, X} <: AbstractFormConstraint
    constraint::C
    context::X
end

struct WrappedFormConstraintNoContext end

"""
    prepare_context(constraint)

Prepares a reusable context for a given form constraint. Returns `WrappedFormConstraintNoContext` by default (i.e. no context needed).
Override this to precompute things that should be shared across multiple [`constrain_form`](@ref) calls.
"""
prepare_context(constraint) = WrappedFormConstraintNoContext()

"""
    constrain_form(wrapped::WrappedFormConstraint, something)

Unwraps the constraint and delegates to [`constrain_form`](@ref) with the inner constraint.
If a context was provided via [`ReactiveMP.prepare_context`](@ref), it is passed as the second argument.
"""
constrain_form(wrapped::WrappedFormConstraint, something) = constrain_form(
    wrapped, wrapped.context, something
)
constrain_form(wrapped::WrappedFormConstraint, ::WrappedFormConstraintNoContext, something) = constrain_form(
    wrapped.constraint, something
)
constrain_form(wrapped::WrappedFormConstraint, context, something) = constrain_form(
    wrapped.constraint, context, something
)

default_form_check_strategy(wrapped::WrappedFormConstraint) = default_form_check_strategy(
    wrapped.constraint
)
default_prod_constraint(wrapped::WrappedFormConstraint) = default_prod_constraint(
    wrapped.constraint
)

"""
    preprocess_form_constraints(constraints)

Converts form constraints into a form compatible with the ReactiveMP inference backend.
A tuple of constraints becomes a [`ReactiveMP.CompositeFormConstraint`](@ref).
Objects that are not subtypes of [`AbstractFormConstraint`](@ref) get wrapped into a [`ReactiveMP.WrappedFormConstraint`](@ref).
"""
function preprocess_form_constraints end

preprocess_form_constraints(constraints::Tuple) = CompositeFormConstraint(
    map(preprocess_form_constraints, constraints)
)
preprocess_form_constraints(constraint::AbstractFormConstraint) = constraint
preprocess_form_constraints(constraint) = WrappedFormConstraint(
    constraint, prepare_context(constraint)
)

"""
    CompositeFormConstraint

A form constraint that chains multiple constraints together, applying them in order via [`constrain_form`](@ref).
Create one by combining constraints with `+` (e.g. `constraint_a + constraint_b`).
All composed constraints must share the same [`default_form_check_strategy`](@ref).
"""
struct CompositeFormConstraint{C} <: AbstractFormConstraint
    constraints::C
end

Base.show(io::IO, constraint::CompositeFormConstraint) = join(
    io, constraint.constraints, " :: "
)

function constrain_form(composite::CompositeFormConstraint, something)
    return reduce(
        (form, constraint) -> constrain_form(constraint, form),
        composite.constraints;
        init = something,
    )
end

function default_prod_constraint(constraint::CompositeFormConstraint)
    return mapfoldl(
        default_prod_constraint, resolve_prod_strategy, constraint.constraints
    )
end

function default_form_check_strategy(composite::CompositeFormConstraint)
    strategies = map(default_form_check_strategy, composite.constraints)
    if !(all(e -> e === first(strategies), TupleTools.tail(strategies)))
        error(
            "Different default form check strategy for composite form constraints found. Use `form_check_strategy` options to specify check strategy.",
        )
    end
    return first(strategies)
end

Base.:+(constraint::AbstractFormConstraint) = constraint

Base.:+(left::AbstractFormConstraint, right::AbstractFormConstraint)   = CompositeFormConstraint((left, right))
Base.:+(left::AbstractFormConstraint, right::CompositeFormConstraint)  = CompositeFormConstraint((left, right.constraints...))
Base.:+(left::CompositeFormConstraint, right::AbstractFormConstraint)  = CompositeFormConstraint((left.constraints..., right))
Base.:+(left::CompositeFormConstraint, right::CompositeFormConstraint) = CompositeFormConstraint((left.constraints..., right.constraints...))
