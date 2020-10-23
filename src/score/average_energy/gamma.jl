
@average_energy(
    form      => Type{ <: Gamma },
    marginals => (q_out::Any, q_α::Any, q_θ::Any),
    meta      => Nothing,
    begin
        return labsgamma(mean(q_α)) + mean(q_α) * log(mean(q_θ)) - (mean(q_α) - 1.0) * log(mean(q_out)) + inv(mean(q_θ)) * mean(q_out)
    end
)
