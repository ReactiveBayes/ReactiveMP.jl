export make_node, rule

using Distributions

function make_node(::typeof(+)) 
    return FactorNode(+, Deterministic, (:out, :in1, :in2), ((1, 2, 3), ), nothing)
end

function make_node(::typeof(+), out, in1, in2)
    node = make_node(+)
    connect!(node, :out, out)
    connect!(node, :in1, in1)
    connect!(node, :in2, in2)
    return node
end

### Out ###

@rule(
    form        => typeof(+),
    edge        => :out,
    vconstraint => Marginalisation,
    messages    => (m_in1::Dirac{T}, m_in2::Dirac{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return Dirac(mean(m_in1) + mean(m_in2))
    end
)

@rule(
    form        => typeof(+),
    edge        => :out,
    vconstraint => Marginalisation,
    messages    => (m_in1::Normal{T}, m_in2::Normal{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return Normal(mean(m_in1) + mean(m_in2), sqrt(var(m_in1) + var(m_in2)))
    end
)

@rule(
    form        => typeof(+),
    edge        => :out,
    vconstraint => Marginalisation,
    messages    => (m_in1::Normal{T}, m_in2::Dirac{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return Normal(mean(m_in1) + mean(m_in2), std(m_in1))
    end
)

@rule(
    form        => typeof(+),
    edge        => :out,
    vconstraint => Marginalisation,
    messages    => (m_in1::Dirac{T}, m_in2::Normal{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return Normal(mean(m_in1) + mean(m_in2), std(m_in2))
    end
)

### In 1 ###

@rule(
    form        => typeof(+),
    edge        => :in1,
    vconstraint => Marginalisation,
    messages    => (m_out::Dirac{T}, m_in2::Dirac{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return Dirac(mean(m_out) - mean(m_in2))
    end
)

@rule(
    form        => typeof(+),
    edge        => :in1,
    vconstraint => Marginalisation,
    messages    => (m_out::Normal{T}, m_in2::Normal{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return Normal(mean(m_out) - mean(m_in2), sqrt(var(m_out) + var(m_in2)))
    end
)

@rule(
    form        => typeof(+),
    edge        => :in1,
    vconstraint => Marginalisation,
    messages    => (m_out::Dirac{T}, m_in2::Normal{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return Normal(mean(m_out) - mean(m_in2), std(m_in2))
    end
)

@rule(
    form        => typeof(+),
    edge        => :in1,
    vconstraint => Marginalisation,
    messages    => (m_out::Normal{T}, m_in2::Dirac{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return Normal(mean(m_out) - mean(m_in2), std(m_out))
    end
)

### In 2 ###

@rule(
    form        => typeof(+),
    edge        => :in2,
    vconstraint => Marginalisation,
    messages    => (m_out::Dirac{T}, m_in1::Dirac{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return Dirac(mean(m_out) - mean(m_in1))
    end
)

@rule(
    form        => typeof(+),
    edge        => :in2,
    vconstraint => Marginalisation,
    messages    => (m_out::Normal{T}, m_in1::Normal{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return Normal(mean(m_out) - mean(m_in1), sqrt(var(m_out) + var(m_in1)))
    end
)

@rule(
    form        => typeof(+),
    edge        => :in2,
    vconstraint => Marginalisation,
    messages    => (m_out::Dirac{T}, m_in1::Normal{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return Normal(mean(m_out) - mean(m_in1), std(m_in1))
    end
)

@rule(
    form        => typeof(+),
    edge        => :in2,
    vconstraint => Marginalisation,
    messages    => (m_out::Normal{T}, m_in1::Dirac{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return Normal(mean(m_out) - mean(m_in1), std(m_out))
    end
)
