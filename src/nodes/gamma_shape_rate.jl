
import StatsFuns: log2π
import SpecialFunctions: loggamma

@node GammaShapeRate Stochastic [out, (α, aliases = [shape]), (β, aliases = [rate])]

@average_energy GammaShapeRate (q_out::Any, q_α::Union{PointMass, SampleList, GammaDistributionsFamily}, q_β::Any) = begin
    mean(loggamma, q_α) - mean(q_α) * mean(log, q_β) - (mean(q_α) - 1.0) * mean(log, q_out) + mean(q_β) * mean(q_out)
end
