@define_factor_node(node = Poisson, type = Stochastic, interfaces = [:out, (:l, aliases = [:λ])])

# ∑ₖ λᵏ log(k!) / k! over k ≥ 1, the term of the energy with no closed form (Evans and Swartz;
# arXiv:1708.06394), approximated, in `R`: the series directly for λ < 50, in BigFloat up to 110,
# and an error beyond, where the estimate is no longer accurate. The series is summed in Float64,
# or BigFloat for a BigFloat `R`, since λᵏ and k! overflow a Float32 long before k = 100.
function approximate_powersum(::Type{R}, l, j = 100) where {R}
    if iszero(l)
        return zero(R)
    elseif l > 110
        error("Cannot compute ∑ [λ^k*log(k!)]/k! for k > $l")
    elseif l < 50 || R === BigFloat
        A = R === BigFloat ? BigFloat : Float64
        s, lk = zero(A), one(A)
        for k in 1:j
            lk *= l
            s += lk * loggamma(A(k + 1)) / gamma(A(k + 1))
        end
        return convert(R, s)
    else
        return convert(R, approximate_powersum(BigFloat, l, 150))
    end
end

@define_average_energy(
    node = Poisson,
    args = (q[:out]::Any, q[:l]::Any),
    body = (args) -> begin
        rest = mean(args.q[:l]) - mean(args.q[:out]) * mean(log, args.q[:l])
        rest + exp(-mean(args.q[:out])) * approximate_powersum(typeof(float(rest)), mean(args.q[:out]))
    end,
)

@define_average_energy(
    node = Poisson,
    args = (q[:out]::PointMass, q[:l]::Any),
    body = (args) -> mean(args.q[:l]) - mean(args.q[:out]) * mean(log, args.q[:l]) + mapreduce(log, +, 1:mean(args.q[:out]); init = 0),
)
