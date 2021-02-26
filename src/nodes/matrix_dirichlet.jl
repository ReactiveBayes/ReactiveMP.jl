import SpecialFunctions: loggamma

@node MatrixDirichlet Stochastic [ out, a ]

@average_energy MatrixDirichlet (q_out::MatrixDirichlet, q_a::PointMass) = begin

    log_mean_marg_out = logmean(q_out)

    H = 0.0
    for k = 1:ndims(q_a)[2] # For all columns
        a_sum = sum(mean(q_a)[:,k])

        H += -loggamma(a_sum) +
        sum(loggamma.(mean(q_a)[:,k])) -
        sum( (mean(q_a)[:,k] .- 1.0).*log_mean_marg_out[:,k] )
    end

    return H
end
