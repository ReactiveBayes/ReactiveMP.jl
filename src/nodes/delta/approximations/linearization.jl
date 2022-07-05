export LinearApproximation
export LinearApproximationKnownInverse
export localLinearization

abstract type LinearApproximation end

struct LinearApproximationKnownInverse{F} <: LinearApproximation
    F_inv::F
end

function localLinearization(g, x_hat::Float64)
    a = ForwardDiff.derivative(g, x_hat)
    b = g(x_hat) - a * x_hat[1]
    return (a, b)
end
