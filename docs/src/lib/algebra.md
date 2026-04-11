# [Algebra utilities](@id lib-helpers-algebra-common)

This page documents linear-algebra building blocks used internally by ReactiveMP.jl's built-in message update rules. They are exposed publicly so that custom rules can reuse them without reimplementing common operations.

## [Matrix constructors](@id lib-helpers-algebra-matrices)

### `diageye` — identity matrix

[`diageye`](@ref) is a convenience alias for constructing a dense identity matrix of a given size. It is equivalent to `Matrix{Float64}(I, n, n)` but reads more clearly in rule code:

```jldoctest
julia> using ReactiveMP; diageye(3)
3×3 Matrix{Float64}:
 1.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  1.0
```

### `CompanionMatrix` — AR coefficient matrix

[`CompanionMatrix`](@ref) represents the companion form of an AR(p) coefficient vector `θ`:

```math
C(\theta) = \begin{bmatrix} \theta_1 & \theta_2 & \cdots & \theta_p \\ 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & & \ddots & \vdots \\ 0 & 0 & \cdots & 0 \end{bmatrix}
```

It is a lazy `AbstractMatrix` — no allocation is made until an element is accessed or a product is computed. Specialized `*` methods exploit the sparse structure for efficient matrix-vector and matrix-matrix products.

This matrix appears in the [Autoregressive node](@ref lib-nodes-ar) and [Continuous transition node](@ref lib-nodes-ctransition) when expressing a higher-order AR process as a first-order state-space model.

### `PermutationMatrix` — structured permutation

[`PermutationMatrix`](@ref) represents an `n×n` permutation matrix as a length-`n` index vector rather than a dense matrix. It supports efficient multiplication with vectors and matrices via specialized `mul!` dispatch, and its inverse is simply its adjoint.

Permutation matrices appear in normalizing flow layers ([`PermutationLayer`](@ref)) to shuffle input dimensions between coupling layers.

### `StandardBasisVector` — sparse one-hot vector

[`StandardBasisVector`](@ref) represents a standard Cartesian basis vector — all zeros except one element — without allocating a dense array. It supports dot products, outer products, and matrix multiplication, all using the sparse structure.

```@docs
diageye
ReactiveMP.CompanionMatrix
ReactiveMP.PermutationMatrix
ReactiveMP.StandardBasisVector
```

## [In-place scalar and array operations](@id lib-helpers-algebra-inplace)

These functions return `alpha * A` or `-A`, **reusing the storage of `A` when the type allows it** (i.e. when `A` is a mutable `Array`). For immutable arrays or scalars they fall back to a regular allocation. This makes them safe to use in rule code regardless of whether the input is mutable.

```@docs
ReactiveMP.mul_inplace!
ReactiveMP.negate_inplace!
```

## [Trace and rank-1 update utilities](@id lib-helpers-algebra-trace)

These functions implement common linear-algebra patterns that appear repeatedly in Gaussian message rules.

### `mul_trace` — allocation-free `tr(A·B)`

[`ReactiveMP.mul_trace`](@ref) computes `tr(A * B)` directly without forming the full product matrix, saving an `O(n²)` allocation for square matrices.

### `rank1update` — `A + x·yᵀ` via BLAS

[`ReactiveMP.rank1update`](@ref) computes `A + x * y'`. For `BlasFloat` element types it dispatches to the BLAS `ger!` routine, which is highly optimized for this pattern.

### `v_a_vT` — `v·a·vᵀ`

[`ReactiveMP.v_a_vT`](@ref) computes `v * a * v'` or `v₁ * a * v₂'`. When `a` is a scalar it avoids forming a temporary matrix. Specialized methods exist for [`StandardBasisVector`](@ref) inputs that exploit the one-hot structure.

```@docs
ReactiveMP.mul_trace
ReactiveMP.rank1update
ReactiveMP.v_a_vT
```

## [Other utilities](@id lib-helpers-algebra-other)

```@docs
ReactiveMP.GammaShapeLikelihood
ReactiveMP.powerset
ReactiveMP.besselmod
ReactiveMP.isonehot
```
