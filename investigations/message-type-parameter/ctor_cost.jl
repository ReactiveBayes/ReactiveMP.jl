# Cost of building a Message/Marginal from an `Any`-typed value (the rule result is inferred Any).
using ReactiveMP, BayesBase, ExponentialFamily, BenchmarkTools
r = Ref{Any}(NormalMeanVariance(1.0, 2.0)); ann = ReactiveMP.AnnotationDict()
direct(r, ann) = Message(r[], false, false, ann)
@noinline make_message(d, ann) = Message(d, false, false, ann)
barrier(r, ann) = make_message(r[], ann)
directq(r, ann) = Marginal(r[], false, false, ann)
println("Message(::Any) inline constructor:  ", @belapsed direct($r, $ann))
println("Message(::Any) through a barrier:   ", @belapsed barrier($r, $ann))
println("Marginal(::Any) inline constructor: ", @belapsed directq($r, $ann))
