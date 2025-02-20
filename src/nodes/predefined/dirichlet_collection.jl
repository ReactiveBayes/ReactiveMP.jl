import SpecialFunctions: loggamma
import Base.Broadcast: BroadcastFunction

@node DirichletCollection Stochastic [out, a]

@average_energy DirichletCollection (q_out::DirichletCollection, q_a::PointMass) = begin
    m_a = mean(q_a)
    logmean = mean(BroadcastFunction(log), q_out)
    # For both matrix and tensor cases, we compute the average energy over all dimensions
    return sum(-loggamma.(sum(m_a, dims = 1)) .+ sum(loggamma.(m_a), dims = 1) .- sum((m_a .- 1.0) .* logmean, dims = 1))
end
