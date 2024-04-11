
@node Bernoulli Stochastic [out, (p, aliases = [θ])]

@average_energy Bernoulli (q_out::Any, q_p::Any) = -mean(q_out) * mean(log, q_p) - (1.0 - mean(q_out)) * mean(mirrorlog, q_p)
