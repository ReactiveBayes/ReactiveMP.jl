export make_node

@node(
    formtype   => Gamma,
    sdtype     => Stochastic,
    interfaces => [ 
        out, 
        (α, aliases = [ shape ]), 
        (θ, aliases = [ scale ])
    ]
)

@average_energy Gamma (q_out::Any, q_α::Any, q_θ::Any) = begin
    return labsgamma(mean(q_α)) + mean(q_α) * log(mean(q_θ)) - (mean(q_α) - 1.0) * log(mean(q_out)) + inv(mean(q_θ)) * mean(q_out)
end
