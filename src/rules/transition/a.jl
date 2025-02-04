
@rule Transition(:a, Marginalisation) (q_out::Union{DiscreteNonParametric, PointMass, Bernoulli}, q_in::Union{DiscreteNonParametric, PointMass, Bernoulli}) = begin
    return DirichletCollection(collect(probvec(q_out)) * probvec(q_in)' .+ 1)
end

@rule Transition(:a, Marginalisation) (q_out_in::Contingency,) = begin
    return DirichletCollection(components(q_out_in) .+ 1)
end

ReactiveMP.rule(
    fform::Type{<:Transition},
    on::Val{:a},
    vconstraint::Marginalisation,
    messages_names::Nothing,
    messages::Nothing,
    marginals_names::Val{m_names} where {m_names},
    marginals::Tuple,
    meta::Any,
    addons::Any,
    ::Any
) = DirichletCollection(components(getdata(first(marginals))) .+ 1), addons
