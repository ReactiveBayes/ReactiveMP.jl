import SpecialFunctions: loggamma

@node MatrixDirichlet Stochastic [ out, a ]

@average_energy MatrixDirichlet (q_out::MatrixDirichlet, q_a::PointMass) = begin
    H = mapreduce(+, zip(eachcol(mean(q_a)), eachcol(logmean(q_out)))) do (q_a_column, logmean_q_out_column)
        return -loggamma(sum(q_a_column)) + sum(loggamma.(q_a_column)) - sum((q_a_column .- 1.0) .* logmean_q_out_column)
    end
    return H
end
