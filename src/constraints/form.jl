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

The supertype of form constraints, which give a product of messages a chosen functional form
(see [`constrain_form`](@ref)). A constraint of another type works too, wrapped by
[`ReactiveMP.preprocess_form_constraints`](@ref) in a [`ReactiveMP.WrappedFormConstraint`](@ref);
subtyping it lets constraints be combined with `+` into a [`CompositeFormConstraint`](@ref).

A constraint implements [`constrain_form`](@ref), [`default_form_check_strategy`](@ref) and
[`default_prod_constraint`](@ref).
"""
abstract type AbstractFormConstraint end

"""
    FormConstraintCheckEach()

The check strategy that applies the form constraint after each pairwise product, inside
[`ReactiveMP.compute_product_of_two_messages`](@ref): `f(f(μ₁ μ₂) μ₃)`. It keeps every
intermediate product in the constrained form, where the product would otherwise grow or lose
its closed form.

See also [`FormConstraintCheckLast`](@ref), [`ReactiveMP.MessageProductContext`](@ref).
"""
struct FormConstraintCheckEach end

function Base.show(io::IO, ::FormConstraintCheckEach)
    print(
        io, get(io, :compact, false) ? "CheckEach" : "FormConstraintCheckEach()"
    )
    return nothing
end

"""
    FormConstraintCheckLast()

The check strategy that applies the form constraint once, to the whole product, at the end of
[`ReactiveMP.compute_product_of_messages`](@ref): `f(μ₁ μ₂ μ₃)`. It is the default of
[`ReactiveMP.MessageProductContext`](@ref), and the cheaper where the intermediate form does not
matter.

See also [`FormConstraintCheckEach`](@ref).
"""
struct FormConstraintCheckLast end

function Base.show(io::IO, ::FormConstraintCheckLast)
    print(
        io, get(io, :compact, false) ? "CheckLast" : "FormConstraintCheckLast()"
    )
    return nothing
end

"""
    FormConstraintCheckPickDefault()

The choice of the constraint's own check strategy, [`default_form_check_strategy`](@ref), for
a caller that resolves it, such as RxInfer when it builds a variable's
[`ReactiveMP.MessageProductContext`](@ref). The engine does not resolve it: the context's
`form_constraint_check_strategy` is [`FormConstraintCheckEach`](@ref) or
[`FormConstraintCheckLast`](@ref), and any other value applies no form constraint.
"""
struct FormConstraintCheckPickDefault end

"""
    default_form_check_strategy(form_constraint)

The check strategy `form_constraint` applies with by default, [`FormConstraintCheckEach`](@ref)
or [`FormConstraintCheckLast`](@ref). A constraint implements it; RxInfer calls it to build a
variable's [`ReactiveMP.MessageProductContext`](@ref), and a [`CompositeFormConstraint`](@ref)
requires its parts to agree.
"""
function default_form_check_strategy end

"""
    default_prod_constraint(form_constraint)

The product strategy for `BayesBase.prod` that `form_constraint` needs, such as
`BayesBase.GenericProd()`, which keeps a product with no closed form as a `ProductOf` for the
constraint to approximate. A constraint implements it; RxInfer calls it to build a variable's
[`ReactiveMP.MessageProductContext`](@ref), and a [`CompositeFormConstraint`](@ref) resolves its
parts' with `BayesBase.resolve_prod_strategy`.
"""
function default_prod_constraint end

"""
    constrain_form(constraint, distribution)
    constrain_form(constraint, context, distribution)

`distribution` in the form `constraint` imposes: the `f` of `q(x) = f(μ₁(x) μ₂(x))`. A
constraint implements it for the distributions it supports; the second form is called for a
[`ReactiveMP.WrappedFormConstraint`](@ref) with the context its
[`ReactiveMP.prepare_context`](@ref) prepared. A constraint that returns its input unchanged keeps
the product's log scale; one that returns something else leaves it undefined.

See also [`AbstractFormConstraint`](@ref), [`ReactiveMP.MessageProductContext`](@ref).
"""
function constrain_form end

"""
    UnspecifiedFormConstraint()

The form constraint that constrains nothing, returning the distribution as it is: the default of
[`ReactiveMP.MessageProductContext`](@ref). Its check strategy is [`FormConstraintCheckLast`](@ref)
and its product strategy `BayesBase.GenericProd()`.
"""
struct UnspecifiedFormConstraint <: AbstractFormConstraint end

default_form_check_strategy(::UnspecifiedFormConstraint) =
    FormConstraintCheckLast()

default_prod_constraint(::UnspecifiedFormConstraint) = GenericProd()

constrain_form(::UnspecifiedFormConstraint, something) = something

"""
    ReactiveMP.WrappedFormConstraint(constraint, context)

A form constraint of a type that does not subtype [`AbstractFormConstraint`](@ref), with the
context its [`ReactiveMP.prepare_context`](@ref) returned. [`ReactiveMP.preprocess_form_constraints`](@ref)
builds it. [`constrain_form`](@ref), [`default_form_check_strategy`](@ref) and
[`default_prod_constraint`](@ref) forward to `constraint`, the first with the context when there
is one.
"""
struct WrappedFormConstraint{C, X} <: AbstractFormConstraint
    constraint::C
    context::X
end

struct WrappedFormConstraintNoContext end

"""
    ReactiveMP.prepare_context(constraint)

A context for `constraint`, computed once when it is wrapped and passed to every
`constrain_form(constraint, context, distribution)` call: a random number generator, a
precomputed quadrature, and so on. By default there is none, and `constrain_form(constraint,
distribution)` is called. A constraint that needs one adds a method.
"""
prepare_context(constraint) = WrappedFormConstraintNoContext()

constrain_form(wrapped::WrappedFormConstraint, something) =
    constrain_form(wrapped, wrapped.context, something)
constrain_form(
    wrapped::WrappedFormConstraint, ::WrappedFormConstraintNoContext, something
) = constrain_form(wrapped.constraint, something)
constrain_form(wrapped::WrappedFormConstraint, context, something) =
    constrain_form(wrapped.constraint, context, something)

default_form_check_strategy(wrapped::WrappedFormConstraint) =
    default_form_check_strategy(wrapped.constraint)
default_prod_constraint(wrapped::WrappedFormConstraint) =
    default_prod_constraint(wrapped.constraint)

"""
    ReactiveMP.preprocess_form_constraints(constraints)

Form constraints as the engine takes them: an [`AbstractFormConstraint`](@ref) as it is, a tuple as
a [`CompositeFormConstraint`](@ref) of its elements, each preprocessed, and any other value
wrapped in a [`ReactiveMP.WrappedFormConstraint`](@ref) with its
[`ReactiveMP.prepare_context`](@ref).
"""
function preprocess_form_constraints end

preprocess_form_constraints(constraints::Tuple) =
    CompositeFormConstraint(map(preprocess_form_constraints, constraints))
preprocess_form_constraints(constraint::AbstractFormConstraint) = constraint
preprocess_form_constraints(constraint) =
    WrappedFormConstraint(constraint, prepare_context(constraint))

"""
    CompositeFormConstraint(constraints::Tuple)

Several form constraints applied in order, the output of each the input of the next. Combine
constraints with `+`, `a + b`, or give a tuple to [`ReactiveMP.preprocess_form_constraints`](@ref).
Its product strategy resolves its parts'.

# Throws

- `ErrorException` from [`default_form_check_strategy`](@ref) when its parts' default check
  strategies differ.
"""
struct CompositeFormConstraint{C} <: AbstractFormConstraint
    constraints::C
end

Base.show(io::IO, constraint::CompositeFormConstraint) =
    join(io, constraint.constraints, " :: ")

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

Base.:+(left::AbstractFormConstraint, right::AbstractFormConstraint) = CompositeFormConstraint((left, right))
Base.:+(left::AbstractFormConstraint, right::CompositeFormConstraint) = CompositeFormConstraint((left, right.constraints...))
Base.:+(left::CompositeFormConstraint, right::AbstractFormConstraint) = CompositeFormConstraint((left.constraints..., right))
Base.:+(left::CompositeFormConstraint, right::CompositeFormConstraint) = CompositeFormConstraint((left.constraints..., right.constraints...))
