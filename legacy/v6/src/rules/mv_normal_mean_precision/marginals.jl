# What remains of MvNormalMeanPrecision here: its two marginal rules for BIFM's
# `TerminalProdArgument`, kept for the BIFM port (Phase 6). The rest is in
# `lib/StandardMessagePassingRules` (Phase 5, step 5).

## TerminalProdArgument / BIFM related

@marginalrule MvNormalMeanPrecision(:out_μ_Λ) (
    m_out::TerminalProdArgument, m_μ::PointMass, m_Λ::PointMass,
) = begin
    return (out = getdist(m_out), μ = m_μ, Λ = m_Λ)
end

@marginalrule MvNormalMeanPrecision(:out_μ_Λ) (
    m_out::PointMass, m_μ::TerminalProdArgument, m_Λ::PointMass,
) = begin
    return (out = m_out, μ = getdist(m_μ), Λ = m_Λ)
end
