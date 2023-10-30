# Extensions and interaction with the Julia ecosystem

`ReactiveMP.jl` exports extra functionality if other Julia packages are loaded in the same environment.

## Optimisers.jl

The [`Optimizers.jl`](https://github.com/FluxML/Optimisers.jl) package defines many standard gradient-based optimisation rules, and tools for applying them to deeply nested models.
The optimizers defined in the `Optimziers.jl` are compatible with the CVI approximation method.

## DiffResults.jl (loaded automatically with the `ForwardDiff.jl`)

The [`DiffResults.jl`](https://github.com/JuliaDiff/DiffResults.jl) provides the `DiffResult` type, which can be passed to in-place differentiation methods instead of an output buffer.
If loaded in the current Julia session enables faster derivatives with the `ForwardDiffGrad` option in the CVI approximation method (in the `Gaussian` case).

