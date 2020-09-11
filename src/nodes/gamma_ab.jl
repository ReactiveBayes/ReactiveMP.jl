export make_node, rule


function make_node(::Type{ <: GammaAB }; factorisation = ((1, 2, 3), ))
    return FactorNode(GammaAB, Stochastic, (:out, :a, :b), factorisation, nothing)
end

function make_node(::Type{ <: GammaAB }, out, a, b; factorisation = ((1, 2, 3), ))
    node = make_node(GammaAB, factorisation = factorisation)
    connect!(node, :out, out)
    connect!(node, :a, a)
    connect!(node, :b, b)
    return node
end

## rules

@rule(
    form        => Type{ <: GammaAB },
    on          => :out,
    vconstraint => Marginalisation,
    messages    => (m_a::Dirac{T}, m_b::Dirac{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return GammaAB(mean(m_a), mean(m_b))
    end
)

## marginalrules 

@marginalrule(
    form        => Type{ <: GammaAB },
    on          => :out_a_b,
    messages    => (m_out::GammaAB{T}, m_a::Dirac{T}, m_b::Dirac{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        q_out = Message(GammaAB(mean(m_a), mean(m_b))) * m_out
        return (getdata(q_out), getdata(m_a), getdata(m_b))
    end
)
