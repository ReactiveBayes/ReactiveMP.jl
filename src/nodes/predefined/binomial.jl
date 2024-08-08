using Distributions
@node Binomial Stochastic [n, k, p]

function expectation_logbinomial(q_k::Binomial, q_n::Binomial)
    f(k, n) = logfactorial(n) - logfactorial(n - k) - logfactorial(k)
    support_k = Distributions.support(q_k)
    support_n = Distributions.support(q_n)

    @assert (support_k ⊆ support_n) error("Support of k $(support_k) should be contained in the support of n $(support_n)")
    return mapreduce(d -> first(d) ≤ last(d) ? f(d...) * pdf(q_k, first(d)) * pdf(q_n, last(d)) : 0, +, product(support_k, support_n))
end

function expectation_logbinomial(q_k::Binomial, q_n::PointMass)
    n = mean(q_n)
    f(k) = logfactorial(n) - logfactorial(n - k) - logfactorial(k)
    support_k = Distributions.support(q_k)

    return mapreduce(d -> f(d) * pdf(q_k, d), +, support_k)
end

function expectation_logbinomial(q_k::PointMass, q_n::Binomial)
    k = mean(q_k)
    f(n) = logfactorial(n) - logfactorial(n - k) - logfactorial(k)
    support_n = Distributions.support(q_n)

    return mapreduce(d -> f(d) * pdf(q_n, d), +, k:maximum(support_n))
end

function expectation_logbinomial(q_k::PointMass, q_n::PointMass)
    k = mean(q_k)
    n = mean(q_n)
    return logfactorial(n) - logfactorial(n - k) - logfactorial(k)
end

@average_energy Binomial (q_n::Union{Binomial, PointMass}, q_k::Union{PointMass, Binomial}, q_p::Union{PointMass, Beta}) = begin
    mean_n = mean(q_n)
    mean_k = mean(q_k)
    logmean_p = mean(log, q_p)
    logmirrormean_p = mean(mirrorlog, q_p)

    term1 = -logmean_p * mean_k - logmirrormean_p * (mean_n - mean_k)
    term2 = -expectation_logbinomial(q_k, q_n)
    return term1 + term2
end
