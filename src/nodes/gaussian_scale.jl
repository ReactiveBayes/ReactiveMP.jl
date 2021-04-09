export GS

import StatsFuns: logπ

struct GS end

@node GS Stochastic [ X, xi ]

@average_energy GS (q_X::ComplexNormal, q_xi::NormalDistributionsFamily) = begin
   
    m_xi, v_xi = mean(q_xi), var(q_xi)
    m_X, v_X = mean(q_X), real(var(q_X))

    m_xi + logπ + exp(-m_xi + v_xi/2)*(abs2(m_X) + v_X)

end