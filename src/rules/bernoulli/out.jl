@rule(
    formtype    => Bernoulli,
    on          => :out,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_p::Any, ),
    meta        => Nothing,
    begin
        rho_1 = exp(logmean(q_p))
        rho_2 = exp(mirroredlogmean(q_p))
        return Bernoulli(rho_1 / (rho_1 + rho_2))
    end
)