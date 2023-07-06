@rule HalfNormal(:out, Marginalisation) (q_v::PointMass,) = begin 
    return Truncated(Normal(0.0, mean(q_v)), 0.0, Inf)
end