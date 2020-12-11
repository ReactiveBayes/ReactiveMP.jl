@rule(
    formtype    => Categorical,
    on          => :out,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_p::Any, ),
    meta        => Nothing,
    begin
        rho = clamp.(exp.(logmean(q_p)), tiny, Inf) # Softens the parameter
        return Categorical(rho ./ sum(rho))
    end
)