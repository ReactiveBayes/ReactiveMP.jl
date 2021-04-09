export rule

@rule GCV(:X, Marginalisation) (q_xi::Any) = begin

    mean_xi, var_xi = mean(q_xi), var(q_xi)

    vX = exp(mean_xi - var_xi/2) .+ 0im

    return ComplexNormal(0, vX, 0)
end