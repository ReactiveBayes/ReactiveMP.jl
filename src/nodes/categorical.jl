
@node Categorical Stochastic [ out, p ]

@average_energy Categorical (q_out::Categorical, q_p::Dirichlet) = -sum(mean(q_out) .* logmean(q_p))