export make_node, score

import SpecialFunctions: digamma

@node(
    formtype   => Dirichlet,
    sdtype     => Stochastic,
    interfaces => [ out, a ]
)

@average_energy Dirichlet (q_out::Dirichlet, q_a::PointMass) = -labsgamma(sum(mean(q_a))) + sum(labsgamma.(mean(q_a))) - sum((mean(q_a) .- one(eltype(mean(q_a)))) .* logmean(q_out))