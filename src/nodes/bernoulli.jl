
@node Bernoulli Stochastic [ out, (p, aliases = [ θ ]) ]

conjugate_type(::Type{ <: Bernoulli }, ::Type{ Val{ :out } }) = Bernoulli
conjugate_type(::Type{ <: Bernoulli }, ::Type{ Val{ :θ } })   = Beta

@average_energy Bernoulli (q_out::Any, q_p::Any) = -mean(q_out) * mean(log, q_p) - (1.0 - mean(q_out)) * mirroredlogmean(q_p)