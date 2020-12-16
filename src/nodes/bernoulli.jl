export make_node, conjugate_type, score

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

@average_energy Bernoulli (q_out::Any, q_p::Any) = -mean(q_out) * logmean(q_p) - (1.0 - mean(q_out)) * mirroredlogmean(q_p)