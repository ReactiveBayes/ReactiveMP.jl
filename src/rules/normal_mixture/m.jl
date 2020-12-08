@rule(
    formtype    => NormalMixture,
    on          => (:m, k),
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_out::Any, q_switch::Any, q_m::NTuple{N1, NormalMeanVariance}, q_p::NTuple{N2, Gamma}) where { N1, N2},
    meta        => Nothing,
    begin
        error("eeeee")
    end
)