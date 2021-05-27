export GaussianScaleSum

import StatsFuns: logπ

struct GaussianScaleSum end

@node GaussianScaleSum Stochastic [ out, s, n ]

@average_energy GaussianScaleSum (q_out::Any, q_s::Any, q_n::Any) = begin
    
    # fetch parameters
    mx, vx = mean(q_out), cov(q_out)
    ms = mean(q_s)
    mn = mean(q_n)
    
    # calculate average energy
    U = logπ + 
        log(exp(ms) + exp(mn)) + 
        (vx + abs2(mx))/(exp(ms) + exp(mn))

    # return average energy
    return U
end

sigmoid(x) = 1 / (1 + exp(-x))