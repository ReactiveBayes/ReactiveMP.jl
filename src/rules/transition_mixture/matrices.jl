@rule TransitionMixture((:matrices, k), Marginalisation) (q_out_in_switch::Contingency,) = begin
    return MatrixDirichlet(components(q_out_in_switch)[k, :, :] .+ 1)
end
