# A Wishart likelihood of G: |G|^(1/2) exp(-E[γ] tr(G S) / 2), so d + 2 degrees of freedom
# and the scale (E[γ] S)⁻¹. A generic-float `cholinv` is symmetric only up to rounding, which
# `Wishart` refuses, hence the `Hermitian`.
scale_matrix_likelihood(d, γ, S) = (V = Matrix(Hermitian(cholinv(γ * S))); Wishart(convert(eltype(V), d + 2), V))

@define_message_update_rule(
    node = MvNormalMeanScaleMatrixPrecision, target = :G,
    args = (q[:out]::Any, q[:μ]::Any, q[:γ]::Any),
    body = (args) -> scale_matrix_likelihood(ndims(args.q[:μ]), mean(args.q[:γ]), difference_moment(args.q[:out], args.q[:μ])),
)

@define_message_update_rule(
    node = MvNormalMeanScaleMatrixPrecision, target = :G,
    args = (q[:out, :μ]::Any, q[:γ]::Any),
    body = (args) -> scale_matrix_likelihood(div(ndims(args.q[:out, :μ]), 2), mean(args.q[:γ]), difference_moment(args.q[:out, :μ])),
)
