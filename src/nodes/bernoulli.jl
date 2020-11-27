export make_node

@node(
    formtype   => Bernoulli,
    sdtype     => Stochastic,
    interfaces => [ 
        out, 
        (p, aliases = [ θ ]), 
    ]
)

conjugate_type(::Type{ <: Bernoulli }, ::Type{ Val{ :out } }) = Bernoulli
conjugate_type(::Type{ <: Bernoulli }, ::Type{ Val{ :θ } })   = Beta

@average_energy(
    formtype  => Bernoulli,
    marginals => (q_out::Any, q_p::Any),
    meta      => Nothing,
    begin
        -mean(q_out) * logmean(q_p) - (1.0 - mean(q_out)) * mirroredlogmean(q_p)
    end
)