using Distributions
@node Binomial Stochastic [n, k, p]

@average_energy Binomial (q_n::Any, q_k::Any, q_p::Any) = begin
    mean_n = mean(q_n)
    mean_k = mean(q_k)
    logmean_p = mean(log, q_p)
    logmirrormean_p = mean(mirrorlog, q_p)

    term1 = -logmean_p * mean_k - logmirrormean_p * (mean_n - mean_k)
    term2 = -expectation_logbinomial(q_k, q_n)
    return term1 + term2
end


function expectation_logbinomial(q_k::Any, q_n::Any)
    f(k, n) = logfactorial(n) - logfactorial(n - k) - logfactorial(k)
    support_k = Distributions.support(q_k)
    support_n = Distributions.support(q_n)

    return mapreduce(d -> first(d) ≤ last(d) ? f(d...) * pdf(q_k, first(d)) * pdf(q_n, last(d)) : 0, +, product(support_k, support_n))
end

function expectation_logbinomial(q_k::Any, q_n::PointMass)
    n = mean(q_n)
    f(k) = logfactorial(n) - logfactorial(n - k) - logfactorial(k)
    support_k = Distributions.support(q_k)

    return mapreduce(d -> d ≤ n ? f(d) * pdf(q_k, d) : 0, +, support_k)
end

function expectation_logbinomial(q_k::PointMass, q_n::Any)
    k = mean(q_k)
    f(n) = logfactorial(n) - logfactorial(n - k) - logfactorial(k)
    support_n = Distributions.support(q_n)

    return mapreduce(d -> d ≥ k ? f(d) * pdf(q_n, d) : 0, +, support_n)
end

function expectation_logbinomial(q_k::PointMass, q_n::PointMass)
    k = mean(q_k)
    n = mean(q_n)
    @assert n ≥ k "n should be larger than k"
    return logfactorial(n) - logfactorial(n - k) - logfactorial(k)
end

