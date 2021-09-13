export PointMassFormConstraint, call_optimizer, call_starting_point

using Distributions
using Optim

"""
    PointMassFormConstraint

One of the form constraint objects. Constraint a message to be in a form of dirac's delta point mass. 
By default uses `Optim.jl` package to find argmin of -logpdf(x). 
Accepts custom `optimizer` callback which might be used to customise optimisation procedure with different packages 
or different arguments for `Optim.jl` package.

See also: [`constrain_form`](@ref), [`DistProduct`](@ref)
"""
struct PointMassFormConstraint{F, P} <: AbstractFormConstraint
    optimizer      :: F
    starting_point :: P   
end

PointMassFormConstraint(; optimizer = default_point_mass_form_constraint_optimizer, starting_point = default_point_mass_form_constraint_starting_point) = PointMassFormConstraint(optimizer, starting_point)

default_form_check_strategy(::PointMassFormConstraint) = FormConstraintCheckLast()

is_point_mass_form_constraint(::PointMassFormConstraint) = true

call_optimizer(pmconstraint::PointMassFormConstraint, data)      = pmconstraint.optimizer(variate_form(data), value_support(data), pmconstraint, data)
call_starting_point(pmconstraint::PointMassFormConstraint, data) = pmconstraint.starting_point(variate_form(data), value_support(data), pmconstraint, data)

function constrain_form(pmconstraint::PointMassFormConstraint, message::Message) 
    data       = ReactiveMP.getdata(message)
    is_clamped = ReactiveMP.is_clamped(message)
    is_initial = ReactiveMP.is_initial(message)
    return Message(call_optimizer(pmconstraint, data), is_clamped, is_initial)
end

function default_point_mass_form_constraint_optimizer(::Type{ Univariate }, ::Type{ Continuous }, constraint::PointMassFormConstraint, distribution)

    target = let distribution = distribution 
        (x) -> -logpdf(distribution, x[1])
    end

    support = Distributions.support(distribution)

    result = if isinf(Distributions.minimum(support)) && isinf(Distributions.maximum(support))
        optimize(target, call_starting_point(constraint, distribution), LBFGS())
    else
        lb = [ Distributions.minimum(support) ]
        rb = [ Distributions.maximum(support) ]
        optimize(target, lb, rb, call_starting_point(constraint, distribution), Fminbox(GradientDescent()))
    end

    if Optim.converged(result)
        return PointMass(Optim.minimizer(result)[1])
    else
        error("Optimisation procedure for point mass estimation did not converge", result)
    end
end

function default_point_mass_form_constraint_starting_point(::Type{ Univariate }, ::Type{ Continuous }, constraint::PointMassFormConstraint, distribution)
    support = Distributions.support(distribution)
    lb      = Distributions.minimum(support)
    rb      = Distributions.maximum(support)
    return if isinf(lb) && isinf(rb)
        return zeros(1)
    else
        error("No default starting point specified for a range [ $(lb), $(rb) ]")
    end
end