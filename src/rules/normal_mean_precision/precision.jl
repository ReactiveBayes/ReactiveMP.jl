@rule(
    form        => Type{ <: NormalMeanPrecision },
    on          => :precision,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_out::Any, q_mean::Any),
    meta        => Nothing,
    begin
        diff = mean(q_out) - mean(q_mean)
        return Gamma(3.0 / 2.0, 2.0 / (var(q_out) + var(q_mean) + diff^2))
    end
)