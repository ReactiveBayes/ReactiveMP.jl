export Algonquin

import StatsFuns: log2π

struct Algonquin end

@node Algonquin Stochastic [ out, s, n, γ ]

@average_energy Algonquin (q_out::Any, q_s::Any, q_n::Any, q_γ::Any) = begin
    
    # fetch parameters
    mx, vx = mean(q_out), cov(q_out)
    ms, vs = mean(q_s), cov(q_s)
    mn, vn = mean(q_n), cov(q_n)
    γ = mean(q_γ)
    
    # calculate average energy
    U = 0.5*(   log2π -
                log(γ) + 
                γ*( mx^2 + vx + log(exp(ms) + exp(mn))^2 + vs*sigmoid(ms-mn)^2 + vn*sigmoid(mn-ms)^2 - 2*mx*log(exp(ms)+exp(mn)) )
    )

    # return average energy
    return U
end

sigmoid(x) = 1 / (1 + exp(-x))