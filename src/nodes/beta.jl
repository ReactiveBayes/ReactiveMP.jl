
import SpecialFunctions: logbeta

@node Beta Stochastic [ out, (a, aliases = [ α ]), (b, aliases = [ β ]) ]

@average_energy Beta (q_out::Any, q_a::Any, q_b::Any) = logbeta(mean(q_a), mean(q_b)) - (mean(q_a) - 1.0) * mean(log, q_out) - (mean(q_b) - 1.0) * mirroredlogmean(q_out)