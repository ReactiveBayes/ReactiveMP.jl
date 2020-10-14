@marginalrule(
    form => Type{ <: MvNormalMeanPrecision },
    on   => :out_μ_Λ,
    messages => (m_out::MvNormalMeanPrecision, m_μ::Dirac, m_Λ::Dirac),
    marginals => Nothing,
    meta => Nothing,
    begin
        q_out = m_out * as_message(MvNormalMeanPrecision(mean(m_μ), mean(m_Λ)))
        return FactorizedMarginal(q_out, m_μ, m_Λ)
    end
)

@marginalrule(
    form => Type{ <: MvNormalMeanPrecision },
    on   => :out_μ_Λ,
    messages => (m_out::Dirac, m_μ::MvNormalMeanPrecision, m_Λ::Dirac),
    marginals => Nothing,
    meta => Nothing,
    begin
        q_μ = m_μ * as_message(MvNormalMeanPrecision(mean(m_out), mean(m_Λ)))
        return FactorizedMarginal(m_out, q_μ, m_Λ)
    end
)

@marginalrule(
    form => Type{ <: MvNormalMeanPrecision },
    on   => :out_μ,
    messages => (m_out::MvNormalMeanPrecision, m_μ::MvNormalMeanPrecision),
    marginals => (q_Λ::Dirac, ),
    meta => Nothing,
    begin
        W_y  = precision(m_out)
        xi_y = W_y * mean(m_out)

        W_m  = precision(m_μ)
        xi_m = W_m * mean(m_μ)

        W_bar = mean(q_Λ)
        
        Λ  = PDMat(Matrix(Hermitian([ W_y + W_bar -W_bar; -W_bar W_m + W_bar ])))
        μ  = inv(Λ) * [ xi_y; xi_m ]
        
        return MvNormalMeanPrecision(μ, Λ)
    end
)