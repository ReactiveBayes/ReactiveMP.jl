@rule HalfNormal(:out, Marginalisation) (q_v::PointMass,) = begin
    mean_v = mean(q_v)
    return Truncated(Normal(zero(eltype(q_v)), sqrt(mean_v)), zero(eltype(q_v)), typemax(float(mean_v)))
end
