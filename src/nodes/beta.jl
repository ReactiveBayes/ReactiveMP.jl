
import SpecialFunctions: logbeta

@node Beta Stochastic [out, α, β]

@average_energy Beta (q_out::Any, q_α::Any, q_β::Any) = logbeta(mean(q_α), mean(q_β)) - (mean(q_α) - 1.0) * mean(log, q_out) - (mean(q_β) - 1.0) * mean(mirrorlog, q_out)
