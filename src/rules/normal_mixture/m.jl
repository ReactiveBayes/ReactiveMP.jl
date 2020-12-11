@rule(
    formtype    => NormalMixture,
    on          => (:m, k),
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_out::Any, q_switch::Any, q_m::NTuple{N1, NormalMeanVariance}, q_p::NTuple{N2, Gamma}) where { N1, N2 },
    meta        => Nothing,
    begin
        z_bar = clamp.(probvec(q_switch), tiny, 1.0 - tiny)
        return NormalMeanVariance(mean(q_out), inv(z_bar[k] * mean(q_p[k])))
    end
)