@marginalrule(
    formtype    => NormalMeanPrecision,
    on          => :out_μ_τ,
    messages    => (m_out::NormalMeanPrecision{T}, m_μ::Dirac{T}, m_τ::Dirac{T}) where { T <: Real },
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return (out = prod(ProdPreserveParametrisation(), NormalMeanPrecision(mean(m_μ), mean(m_τ)), m_out), μ = m_μ, τ = m_τ)
    end
)

@marginalrule(
    formtype    => NormalMeanPrecision,
    on          => :out_μ_τ,
    messages    => (m_out::Dirac{T}, m_μ::NormalMeanPrecision{T}, m_τ::Dirac{T}) where { T <: Real },
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return (out = m_out, μ = prod(ProdPreserveParametrisation(), m_μ, NormalMeanPrecision(mean(m_out), mean(m_τ))), τ = m_τ)
    end
)

@marginalrule(
    formtype    => NormalMeanPrecision,
    on          => :out_μ,
    messages    => (m_out::NormalMeanPrecision{T}, m_μ::NormalMeanPrecision{T}) where { T <: Real },
    marginals   => (q_τ::Gamma{T}, ),
    meta        => Nothing,
    begin
        W_out  = invcov(m_out)
        W_μ    = invcov(m_μ)
        xi_out = W_out * mean(m_out)
        xi_μ   = W_μ * mean(m_μ)

        W_bar = mean(q_τ)

        W = [ W_out + W_bar -W_bar; -W_bar W_μ + W_bar ]
        m = cholinv(W) * [ xi_out; xi_μ ]

        return MvNormalMeanPrecision(m, W)
    end
)