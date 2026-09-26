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

## Pages

- [BIFM](@ref page-bifm): the time slice, its algorithm [`BIFMSmoother`](@ref), and the
  equations of both passes.
- [BIFMHelper](@ref page-bifm-helper): the node at the start of the chain.
