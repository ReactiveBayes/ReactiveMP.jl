"""
    FormConstraintsGenerator

This structure contains a `callback` that creates a single independent `form_constraint` on each invocation.
Used to assign form constraints for arrays of random variables such that each random variable get its unique copy of the `form_constraint` object.
Makes it easier to use local cache structures in form constraints that should not be shared between multiple random variables.

# Arguments 
- `form_constraint_callback`: a function or a callable object `() -> ...` that accepts no arguments and returns a unique instance of some form constraint
- `form_constraint_repr`: a string that describes what form constraint will be created upon calling the `form_constraint_callback`

See also: [`AbstractFormConstraint`](@ref)
"""
struct FormConstraintsGenerator{F}
    form_constraint_callback :: F
    form_constraint_repr     :: String
end

as_form_constraint(generator::FormConstraintsGenerator) = generator.form_constraint_callback()

Base.show(io::IO, generator::FormConstraintsGenerator) = print(io, "foreach(", generator.form_constraint_repr, ")")

default_prod_constraint(FormConstraintsGenerator) = nothing