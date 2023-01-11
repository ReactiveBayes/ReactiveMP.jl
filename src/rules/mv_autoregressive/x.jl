
@rule MAR(:x, Marginalisation) (m_y::MultivariateNormalDistributionsFamily, q_a::MultivariateNormalDistributionsFamily, q_Λ::Any, meta::MARMeta) =
begin
    ma, Va = mean_cov(q_a)
    my, Vy = mean_cov(m_y)

    mΛ = mean(q_Λ)

    order, ds = getorder(meta), getdimensionality(meta)
    dim = order*ds

    mA = mar_companion_matrix(order, ds, ma)
    mW = mar_transition(getorder(meta), mΛ)
    # this should be inside MARMeta
    es = [uvector(dim, i) for i in 1:ds]
    Fs = [mask_mar(order, ds, i) for i in 1:ds]

    Λ = sum(sum(es[j]'*mW*es[i]*Fs[j]*Va*Fs[i]' for i in 1:ds) for j in 1:ds)

    Σ₁ = Hermitian(pinv(mA)*(Vy)*pinv(mA') + pinv(mA'*mW*mA))

    Ξ = (pinv(Σ₁) + Λ)
    z = pinv(Σ₁)*pinv(mA)*my

    return MvNormalWeightedMeanPrecision(z, Ξ)
end

@rule MAR(:x, Marginalisation) (q_y::MultivariateNormalDistributionsFamily, q_a::MultivariateNormalDistributionsFamily, q_Λ::Any, meta::MARMeta) = begin

    ma, Va = mean_cov(q_a)
    my, Vy = mean_cov(q_y)

    mΛ = mean(q_Λ)

    order, ds = getorder(meta), getdimensionality(meta)
    dim = order*ds

    mA = mar_companion_matrix(order, ds, ma)
    mW = mar_transition(getorder(meta), mΛ)
    
    # this should be inside MARMeta
    es = [uvector(dim, i) for i in 1:ds]
    Fs = [mask_mar(order, ds, i) for i in 1:ds]

    Λ = sum(sum(es[j]'*mW*es[i]*Fs[j]*Va*Fs[i]' for i in 1:ds) for j in 1:ds)
    Λ₀ = Hermitian(mA'*mW*mA)

    Ξ = Λ₀ + Λ
    z = Λ₀*pinv(mA)*my

    return MvNormalWeightedMeanPrecision(z, Ξ)
end