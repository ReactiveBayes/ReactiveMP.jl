export Poisson

@node Poisson Stochastic [out, (l, aliases = [λ])]

# ∑ [λ^k*log(k!)]/k! from k=0 to inf
# Approximates the above sum for calculation of averageEnergy and differentialEntropy
# @ref https://arxiv.org/pdf/1708.06394.pdf
function apprSum(l, j=100)
    sum([(l)^(k)*logfactorial(k)/exp(logfactorial(k)) for k in collect(0:j)])
end

@average_energy Poisson (q_out::Any, q_l::Any) =
    mean(q_l) - mean(q_out)*mean(log, q_l) + exp(-mean(q_out))*apprSum(mean(q_out))