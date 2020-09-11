@rule(
    form        => Type{ <: NormalMeanPrecision },
    on          => :precision,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_out::Any, q_mean::Any),
    meta        => Nothing,
    begin
        diff = mean(marginals[1]) - mean(marginals[2])
        return GammaAB(3.0 / 2.0, 1.0 / 2.0 * (var(marginals[2]) + var(marginals[1]) + diff^2))
    end
)