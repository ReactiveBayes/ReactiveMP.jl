export Uninformative

struct Uninformative end

@node Uninformative Stochastic [out]

@average_energy Uninformative (q_out::Any,) = entropy(q_out)
