@rule GammaMixture((:b, k), Marginalisation) (q_out::Any, q_switch::Any, q_a::Any) = begin
    π_k = probvec(q_switch)[k]
    return GammaShapeRate(1 + π_k * mean(q_a), π_k * mean(q_out))
end
