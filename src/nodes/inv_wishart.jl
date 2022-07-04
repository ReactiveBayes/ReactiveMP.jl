
import StatsFuns: logπ
import SpecialFunctions: loggamma
import Distributions: InverseWishart

@node InvWishart Stochastic [out, (ν, aliases = [df]), (S, aliases = [scale])]

@average_energy InvWishart (q_out::Any, q_ν::PointMass, q_S::Any) = begin
    d = dim(q_out)

    m_q_ν   = mean(q_ν)

    return 0.5 * (
        m_q_ν * (-mean(logdet, S) + d * log(2)) -
        mean(logdet, q_out) * (m_q_ν + d + 1.0) +
        tr(mean(q_S) * mean(inv, q_out)) + 0.5 * d * (d - 1) * logπ
    ) + mapreduce(i -> loggamma(0.5 * (m_q_ν + 1.0 - i)), +, 1:d)
end
