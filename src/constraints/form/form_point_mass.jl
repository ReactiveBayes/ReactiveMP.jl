export PointMassFormConstraint

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
struct PointMassFormConstraint{F}
    optimizer :: F
end

PointMassFormConstraint() = PointMassFormConstraint(default_point_mass_form_constraint_optimizer)

default_form_check_strategy(::PointMassFormConstraint) = FormConstraintCheckLast()

function constrain_form(pmconstraint::PointMassFormConstraint, message::Message) 
    data       = ReactiveMP.getdata(message)
    is_clamped = ReactiveMP.is_clamped(message)
    is_initial = ReactiveMP.is_initial(message)
    return Message(pmconstraint.optimizer(variate_form(data), value_support(data), data), is_clamped, is_initial)
end

function default_point_mass_form_constraint_optimizer(::Type{ Univariate }, ::Type{ Continuous }, distribution)

    target = let distribution = distribution 
        (x) -> -logpdf(distribution, x[1])
    end

    support = Distributions.support(distribution)

    result = if isinf(Distributions.minimum(support)) && isinf(Distributions.maximum(support))
        optimize(target, zeros(1), LBFGS())
    else
        error("TODO optimise function in point mass constraint")
    end

    if Optim.converged(result)
        return PointMass(Optim.minimizer(result)[1])
    else
        error("Optimisation procedure for point mass estimation did not converge", result)
    end
end