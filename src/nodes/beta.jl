
@node Beta Stochastic [ out, (a, aliases = [ α ]), (b, aliases = [ β ]) ]

@average_energy Beta (q_out::Any, q_a::Any, q_b::Any) = labsbeta(mean(q_a), mean(q_b)) - (mean(q_a) - 1.0) * logmean(q_out) - (mean(q_b) - 1.0) * mirroredlogmean(q_out)