
struct FormConstraintSpecification{C, P}
    form_constraint :: C
    prod_constraint :: P
end

FormConstraintSpecification(form_constraint) = FormConstraintSpecification(form_constraint, default_prod_constraint(form_constraint))

Base.show(io::IO, spec::FormConstraintSpecification) = print(io, spec.form_constraint, " [ prod_constraint = ", spec.prod_constraint, "]")