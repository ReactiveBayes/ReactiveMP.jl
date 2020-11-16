export make_node

@node(
    formtype   => Beta,
    sdtype     => Stochastic,
    interfaces => [ 
        out, 
        (a, aliases = [ α ]), 
        (b, aliases = [ β ])
    ]
)

@average_energy(
    formtype  => Beta,
    marginals => (q_out::Any, q_a::Any, q_b::Any),
    meta      => Nothing,
    begin
        labsbeta(mean(q_a), mean(q_b)) - (mean(q_a) - 1.0) * logmean(q_out) - (mean(q_b) - 1.0) * mirroredlogmean(q_out)
    end
)