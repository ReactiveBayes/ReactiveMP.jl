import Base.Broadcast: BroadcastFunction

@node Categorical Stochastic [out, p]

@average_energy Categorical (q_out::Union{Categorical, Multinomial}, q_p::Any) = -sum(probvec(q_out) .* mean(BroadcastFunction(clamplog), q_p))
