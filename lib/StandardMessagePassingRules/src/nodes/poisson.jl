@define_factor_node(node = Poisson, type = Stochastic, interfaces = [:out, (:l, aliases = [:λ])])

# ∑ₖ λᵏ log(k!) / k! over k ≥ 1, the term of the energy with no closed form (Evans and Swartz;
# arXiv:1708.06394), approximated: the series directly for λ < 50, in BigFloat up
# to 110, and an error beyond, where the estimate is no longer accurate.
function approximate_powersum(::Type{R}, l::T, j = 100) where {R, T}
    if l == zero(T)
        return zero(T)
    elseif l > 110
        error("Cannot compute ∑ [λ^k*log(k!)]/k! for k > $l")
    elseif l < 50 || R === BigFloat
        s, lk = zero(R), one(R)
        for k in 1:j
            lk *= l
            s += lk * loggamma(k + 1) / gamma(k + 1)
        end
        return s
    else
        return convert(T, approximate_powersum(BigFloat, l, 150))
    end
end

@define_average_energy(
    node = Poisson,
    args = (q[:out]::Any, q[:l]::Any),
    body = (args) -> mean(args.q[:l]) - mean(args.q[:out]) * mean(log, args.q[:l]) +
        exp(-mean(args.q[:out])) * approximate_powersum(Float64, mean(args.q[:out])),
)

@define_average_energy(
    node = Poisson,
    args = (q[:out]::PointMass, q[:l]::Any),
    body = (args) -> mean(args.q[:l]) - mean(args.q[:out]) * mean(log, args.q[:l]) + mapreduce(log, +, 1:mean(args.q[:out]); init = 0),
)
