import SpecialFunctions: digamma, loggamma
import Base.Broadcast: BroadcastFunction

@node Dirichlet Stochastic [out, a]

@average_energy Dirichlet (q_out::Dirichlet, q_a::PointMass) =
    -loggamma(sum(mean(q_a))) + sum(loggamma.(mean(q_a))) - sum((mean(q_a) .- 1.0) .* mean(BroadcastFunction(log), q_out))
