export rule

using SpecialFunctions: logfactorial

@rule Categorical(:p, Marginalisation) (q_out::Any,) = begin
    return Dirichlet(probvec(q_out) .+ one(eltype(probvec(q_out))))
end

@rule Categorical(:p, Marginalisation) (q_out::PointMass,) = begin
    @logscale -logfactorial(length(probvec(q_out)))
    return Dirichlet(probvec(q_out) .+ one(eltype(probvec(q_out))))
end