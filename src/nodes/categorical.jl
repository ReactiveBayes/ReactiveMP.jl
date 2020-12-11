export make_node

@node(
    formtype   => Categorical,
    sdtype     => Stochastic,
    interfaces => [ out, p ]
)

@average_energy(
    formtype  => Categorical,
    marginals => (q_out::Categorical, q_p::Dirichlet),
    meta      => Nothing,
    begin
        -sum(mean(q_out) .* logmean(q_p))
    end
)