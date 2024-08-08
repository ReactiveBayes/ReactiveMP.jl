using SpecialFunctions, LogExpFunctions
N = 1000
K = 40
ss = (identity, (x -> -logfactorial(x - k) for k in 0:K)...)
p = 10rand(K + 1)
θ = [-1.6log(1 - 0.9); p ./ sum(p)]
lpdf(x) = mapreduce((s, t) -> s(x) * t, +, ss, θ)
normalization = logsumexp(map(lpdf, collect(K:(N + 50))))
pdf(x) = exp(lpdf(x) - normalization)
using Plots
scatter(collect(K:100), pdf.(K:100))

struct StrangeDistribution
    θ
    N
    K
end
using Plots
using ReactiveMP, BayesBase, Distributions, LogExpFunctions
using ExponentialFamily
import ForwardDiff: derivative

basemeasure = n -> ReactiveMP.expectation_logbinomial(Binomial(5, 0.3), PointMass(n))
ss = (identity,)
η = (mean(mirrorlog, Beta(10, 300)),)
supp = 10:200
A(η) = logsumexp(map(x -> mapreduce((f, θ) -> θ' * f(x) + log(basemeasure(x)), collect, ss, η), supp))
attributes = ExponentialFamilyDistributionAttributes(basemeasure, ss, A, supp)
ef = ExponentialFamilyDistribution(Univariate, η, nothing, attributes)

inputs = [(Binomial(20, 0.1), PointMass(0.001)), (Binomial(20, 0.8), Beta(20.5, 18.3)), (Binomial(10, 0.8), Beta(0.8, 3.3)), (Binomial(5, 0.01), Beta(20.5, 18.3))]

binom_approximate = []
output = []
for input in inputs
    η = (mean(log, last(input)) - mean(mirrorlog, last(input)),)
    push!(binom_approximate, convert(Binomial, ExponentialFamilyDistribution(Binomial, η, maximum(support(first(input))))))
    push!(output, @call_rule Binomial(:n, Marginalisation) (q_k = first(input), q_p = last(input)))

    # mean_output = derivative(x -> getlogpartition(output)(x), η[1])
end
getsupport(output[1])
using StatsPlots
i = 3
scatter(collect(getsupport(output[i])), pdf(output[i], getsupport(output[i])))

k = Binomial(30, 0.999)
n = Binomial(70, 0.4)
output = @call_rule Binomial(:p, Marginalisation) (m_k = k, m_n = n)
vmp_output = @call_rule Binomial(:p, Marginalisation) (q_k = k, q_n = n)
lp = mean(map(x -> exp(logpdf(output, x)), collect(0:0.0001:1)))
pf = x -> exp(logpdf(output, x)) / lp
dt = 0.0001
entropy_sum_product = mean(map(x -> pf(x) * log(pf(x)), collect(0:dt:1)))
entropy(vmp_output)
plot(collect(0:0.0001:1), pf.(collect(0:0.0001:1)), label = "sum product")
plot!(vmp_output, label = "vmp")
