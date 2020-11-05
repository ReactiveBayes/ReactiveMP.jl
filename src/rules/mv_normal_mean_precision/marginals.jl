@marginalrule(
    formtype  => MvNormalMeanPrecision,
    on        => :out_μ_Λ,
    messages  => (m_out::MvNormalMeanPrecision, m_μ::Dirac, m_Λ::Dirac),
    marginals => Nothing,
    meta      => Nothing,
    begin
        return (prod(ProdPreserveParametrisation(), MvNormalMeanPrecision(mean(m_μ), mean(m_Λ)), m_out), m_μ, m_Λ)
    end
)

@marginalrule(
    formtype  => MvNormalMeanPrecision,
    on        => :out_μ_Λ,
    messages  => (m_out::Dirac, m_μ::MvNormalMeanPrecision, m_Λ::Dirac),
    marginals => Nothing,
    meta      => Nothing,
    begin
        return (m_out, prod(ProdPreserveParametrisation(), m_μ, MvNormalMeanPrecision(mean(m_out), mean(m_Λ))), m_Λ)
    end
)

@marginalrule(
    formtype  => MvNormalMeanPrecision,
    on        => :out_μ,
    messages  => (m_out::MvNormalMeanPrecision, m_μ::MvNormalMeanPrecision),
    marginals => (q_Λ::Any, ),
    meta      => Nothing,
    begin
        W_y  = invcov(m_out)
        xi_y = W_y * mean(m_out)

        W_m  = invcov(m_μ)
        xi_m = W_m * mean(m_μ)

        W_bar = mean(q_Λ)
        
        Λ  = [ W_y + W_bar -W_bar; -W_bar W_m + W_bar ]
        μ  = cholinv(Λ) * [ xi_y; xi_m ]
        
        return MvNormalMeanPrecision(μ, Λ)
    end
)