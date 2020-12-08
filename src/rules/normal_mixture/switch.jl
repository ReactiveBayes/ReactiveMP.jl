@rule(
    formtype    => NormalMixture,
    on          => :switch,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_out::Any, q_m::NTuple{N, NormalMeanVariance}, q_p::NTuple{N, Gamma}) where { N },
    meta        => Nothing,
    begin
        error(1)
    end
)