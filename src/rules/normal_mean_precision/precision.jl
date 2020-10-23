@rule(
    form        => Type{ <: NormalMeanPrecision },
    on          => :τ,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_out::Any, q_μ::Any),
    meta        => Nothing,
    begin
        diff = mean(q_out) - mean(q_μ)
        return Gamma(3.0 / 2.0, 2.0 / (var(q_out) + var(q_μ) + diff^2))
    end
)