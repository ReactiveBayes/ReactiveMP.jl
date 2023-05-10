module ReactiveMPDiffResultsExt

using ReactiveMP, DiffResults

# Enable fast path in the CVI approximation method for the Gaussian case
function ReactiveMP.compute_df_mv(::CVI{R, O, ForwardDiffGrad}, logp::F, vec::AbstractVector) where {R, O, F}
    result = DiffResults.HessianResult(vec)
    result = ForwardDiff.hessian!(result, logp, vec)
    return DiffResults.gradient(result), DiffResults.hessian(result) ./ 2
end

end
