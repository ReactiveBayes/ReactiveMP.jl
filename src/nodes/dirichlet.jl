export make_node

import SpecialFunctions: digamma

@node(
    formtype   => Dirichlet,
    sdtype     => Stochastic,
    interfaces => [ out, a ]
)

@average_energy(
    formtype  => Dirichlet,
    marginals => (q_out::Dirichlet, q_a::Dirac),
    meta      => Nothing,
    begin
        -labsgamma(sum(mean(q_a))) + sum(labsgamma.(mean(q_a))) - sum((mean(q_a) .- one(eltype(mean(q_a)))) .* logmean(q_out))
    end
)