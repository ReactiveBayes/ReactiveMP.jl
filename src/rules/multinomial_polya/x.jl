function logistic_stic_breaking(m)
    Km1 = length(m)

    p = Array{Float64}(undef, Km1+1)
    p[1] = logistic(m[1])
    for i in 2:Km1
        p[i] = logistic(m[i])*(1 - sum(p[1:i-1]))
    end
    p[end] = 1 - sum(p[1:end-1])
    return p
end

@rule MultinomialPolya(:x, Marginalisation) (q_N::PointMass, q_ψ::GaussianDistributionsFamily, meta::Union{MultinomialPolyaMeta, Nothing}) = begin
    N = mean(q_N)
    m_ψ = mean(q_ψ)
    p = logistic_stic_breaking(m_ψ)
    return Multinomial(N, p)
end
