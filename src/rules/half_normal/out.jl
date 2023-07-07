@rule HalfNormal(:out, Marginalisation) (q_v::PointMass,) = begin
    return Truncated(Normal(0.0, sqrt(mean(q_v))), 0.0, Inf)
end
