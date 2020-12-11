@rule(
    formtype    => Categorical,
    on          => :p,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_out::Any, ),
    meta        => Nothing,
    begin
        return Dirichlet(probvec(q_out) .+ one(eltype(probvec(q_out))))
    end
)