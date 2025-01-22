import SpecialFunctions: loggamma
import Base.Broadcast: BroadcastFunction

@node TensorDirichlet Stochastic [out, a]

@average_energy TensorDirichlet (q_out::TensorDirichlet, q_a::PointMass) = begin
    m_a = mean(q_a)
    logmean = mean(BroadcastFunction(log), q_out)
    return sum(-loggamma.(sum(m_a, dims = 1)) .+ sum(loggamma.(m_a), dims = 1) .- sum((m_a .- 1.0) .* logmean, dims = 1))
end
