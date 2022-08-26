
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

resolve_marginal_form_prod(constraints, model, name) = resolve_form_prod(constraints, model, constraints.marginalsform, name)
resolve_messages_form_prod(constraints, model, name) = resolve_form_prod(constraints, model, constraints.messagesform, name)

# Preoptimised dispatch rule for empty form constraints
resolve_form_prod(constraints, model, ::NamedTuple{()}, name) = (nothing, nothing)

function resolve_form_prod(constraints, model, specification, name)
    entry = get(specification, name, __EmptyFormConstraintSpecification)
    return entry.form_constraint, entry.prod_constraint
end
