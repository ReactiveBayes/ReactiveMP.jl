using Distributions
@node Binomial Stochastic [n, k, p] 

function expectation_klogfactorialnmk(q_k::Binomial,q_n::Binomial)
    f(k,n) = k * logfactorial(n-k)

    support_k = Distributions.support(q_k)
    support_n = Distributions.support(q_n)

    @assert (support_k ⊆ support_n) error("Support of k $(support_k) should be contained in the support of n $(support_n)")
    return mean(map(d -> first(d) ≤ last(d) ?  f(d...) : 0, product(support_k, support_n)))
end

function expectation_nlogfactorialnmk(q_k::Binomial,q_n::Binomial)
    f(k,n) = n * logfactorial(n-k)

    support_k = Distributions.support(q_k)
    support_n = Distributions.support(q_n)

    @assert (support_k ⊆ support_n) error("Support of k $(support_k) should be contained in the support of n $(support_n)")
    return mean(map(d -> first(d) ≤ last(d) ?  f(d...) : 0, product(support_k, support_n)))
end

function expectation_klogfactorial(q_k::Binomial)
    support_k = Distributions.support(q_k)
    return mean(map(k -> k*logfactorial(k), support_k))
end

function expectation_logfactorial(q_k::Binomial)
    support_k = Distributions.support(q_k)
    return mean(map(logfactorial, support_k))
end


@average_energy Binomial (q_n::Any, q_k::Any, q_p::Any) = begin
    mean_n   = mean(q_n)
    mean_k   = mean(q_k)
    logmean_p = mean(log, q_p)
    logmirrormean_p = mean(mirrorlog, q_p)

    term1 = -logmean_p*(mean_k*expectation_logfactorial(q_n) - expectation_klogfactorialnmk(q_k, q_n) - expectation_klogfactorial(q_k))

    term2 = -logmirrormean_p*((expectation_klogfactorial(q_n) - mean_k*expectation_logfactorial(q_n)) - expectation_nlogfactorialnmk(q_k, q_n) +expectation_klogfactorialnmk(q_k, q_n) - mean_n*expectation_logfactorial(q_k) + expectation_klogfactorial(q_k))

    return term1 + term2
end
