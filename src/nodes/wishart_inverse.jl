export InverseWishart

import StatsFuns: logπ
import SpecialFunctions: loggamma
import Distributions: InverseWishart

@node InverseWishart Stochastic [out, (ν, aliases = [df]), (S, aliases = [scale])]

@average_energy InverseWishart (q_out::Any, q_ν::PointMass, q_S::Any) = begin
    d = size(q_out, 1)

    m_q_ν = mean(q_ν)

    return 0.5 * (m_q_ν * (-mean(logdet, q_S) + d * log(2)) + mean(logdet, q_out) * (m_q_ν + d + 1) + tr(mean(q_S) * mean(cholinv, q_out)) + d * (d - 1) / 2 * logπ) +
           mapreduce(i -> loggamma((m_q_ν + 1 - i) / 2), +, 1:d)
end
