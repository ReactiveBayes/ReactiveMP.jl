export Poisson

@node Poisson Stochastic [out, (l, aliases = [λ])]

# ∑ [λ^k*log(k!)]/k! from k=0 to inf
# Approximates the above sum for calculation of averageEnergy and differentialEntropy
# @ref https://arxiv.org/pdf/1708.06394.pdf
function __approximate_powersum(::Type{R}, l::T, j = 100) where {R, T}
    if l == zero(T)
        return zero(T)
    elseif l > 110
        # We give up on l>110, otherwise estimates are quite inaccurate
        error("Cannot compute ∑ [λ^k*log(k!)]/k! for k > $l")
    elseif l < 50 || R === BigFloat # asymptotic, does not work for large `l`
        s = zero(R)
        lk = one(R)
        for k in 1:j
            lk *= l
            s += lk * loggamma(k + 1) / gamma(k + 1)
        end
        return s
    else
        # Try `BigFloat`
        return convert(T, __approximate_powersum(BigFloat, l, 150))
    end
end

@average_energy Poisson (q_out::Any, q_l::Any) = mean(q_l) - mean(q_out) * mean(log, q_l) + exp(-mean(q_out)) * __approximate_powersum(Float64, mean(q_out))

@average_energy Poisson (q_out::PointMass, q_l::Any) = mean(q_l) - mean(q_out) * mean(log, q_l) + mapreduce(log, +, (1:mean(q_out)); init=0)
