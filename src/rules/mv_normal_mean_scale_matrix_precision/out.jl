# Variational                       # 
# --------------------------------- #
@rule MvNormalMeanScaleMatrixPrecision(:out, Marginalisation) (q_μ::Any, q_γ::Any, q_G::Any) = MvNormalMeanPrecision(mean(q_μ), mean(q_γ) * mean(q_G))

@rule MvNormalMeanScaleMatrixPrecision(:out, Marginalisation) (m_μ::MultivariateNormalDistributionsFamily, q_γ::Any, q_G::Any) = begin
    μ_bar, Λ_μ = mean_precision(m_μ)
    Λ_f = mean(q_γ) * mean(q_G)
    
    # Step 1: form M = Λ_μ + Λ_f
    M = Λ_μ + Λ_f

    # Step 2: factorize M (Cholesky since M is PD)
    F = fastcholesky(M)

    # Step 3: solve M X = Λ_μ  (multiple RHS solve)
    X = F \ Λ_μ

    # Step 4: form Λ_out
    Λ_out = Λ_μ - Λ_μ * X

    return MvNormalMeanPrecision( μ_bar, Λ_out )
end
