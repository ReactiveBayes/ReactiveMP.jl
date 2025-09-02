# Variational                       # 
# --------------------------------- #
@rule MvNormalMeanScaleMatrixPrecision(:μ, Marginalisation) (q_out::Any, q_γ::Any, q_G::Any) = MvNormalMeanPrecision(mean(q_out), mean(q_γ) * mean(q_G))

@rule MvNormalMeanScaleMatrixPrecision(:μ, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, q_γ::Any, q_G::Any) = begin
    out_bar, Λ_out = mean_precision(m_out)
    Λ_f = mean(q_γ) * mean(q_G)
    
    # Step 1: form M = Λ_out + Λ_f
    M = Λ_out + Λ_f

    # Step 2: factorize M (Cholesky since M is PD)
    F = fastcholesky(M)

    # Step 3: solve M X = Λ_out  (multiple RHS solve)
    X = F \ Λ_out

    # Step 4: form Λ_μ
    Λ_μ = Λ_out - Λ_out * X

    return MvNormalMeanPrecision( out_bar, Λ_μ )
end