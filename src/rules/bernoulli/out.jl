@rule(
    formtype    => Bernoulli,
    on          => :out,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_p::Any, ),
    meta        => Nothing,
    begin
        rho_1 = clamp(exp(logmean(q_p)), tiny, huge)
        rho_2 = clamp(exp(mirroredlogmean(q_p)), tiny, huge)
        return Bernoulli(rho_1 / (rho_1 + rho_2))
    end
)