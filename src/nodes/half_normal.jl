export HalfNormal

struct HalfNormal end

@node HalfNormal Stochastic [out, (v, aliases = [var, σ²])]

@average_energy HalfNormal (q_out::Any, q_v::Any) = begin
    out_mean, out_var = mean_var(q_out)
    return (log(π / 2) + mean(log, q_v) + mean(inv, q_v) * (out_mean^2 + out_var)) / 2
end
