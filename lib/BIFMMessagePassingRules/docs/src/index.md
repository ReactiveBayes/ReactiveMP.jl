# BIFMMessagePassingRules

Backward-information-filter forward-marginal (BIFM) smoothing of a linear state-space model, as
two nodes: [`BIFM`](@ref), a whole time slice, and [`BIFMHelper`](@ref), which starts the chain.
Use it to smooth a long linear Gaussian chain with known matrices in two passes, without
inverting a covariance at every step.

!!! warning "No free energy"
    The free energy of a model with BIFM is not supported. Asking for it throws a
    [`BIFMFreeEnergyError`](@ref BIFMMessagePassingRules.BIFMFreeEnergyError); run the
    inference without it.

```@docs
BIFMMessagePassingRules
```

!!! info "Where these rules run"
    This package defines message passing rules; it does not build or run models. The
    [ReactiveMP](https://reactivebayes.github.io/ReactiveMP.jl/dev/) engine runs the rules on a
    factor graph, and [RxInfer](https://github.com/ReactiveBayes/RxInfer.jl) builds that graph from
    a model written with [GraphPPL](https://github.com/ReactiveBayes/GraphPPL.jl). The examples
    here call the rules directly, as a test or an interactive session does.

## Pages

- [BIFM](@ref page-bifm): the time slice, its algorithm [`BIFMSmoother`](@ref), and the
  equations of both passes.
- [BIFMHelper](@ref page-bifm-helper): the node at the start of the chain.
