"""
Return local linearization of g around expansion point x_hat
for Delta node with single input interface
"""
function localLinearizationSingleIn(g::Function, x_hat::Float64)
    a = ForwardDiff.derivative(g, x_hat)
    b = g(x_hat) - a*x_hat

    return (a, b)
end

@rule DeltaFn{f}(:out, Marginalisation) (m_out::Any, m_ins::NTuple{N, Any}, meta::DeltaExtended{T}) where { f, N, T } = begin
    return NormalMeanPrecision(f(mean.(m_ins)...), 1.0)
end


@rule DeltaFn{f}(:out, Marginalisation) (m_out::Any, m_in::NTuple{1, Any}, meta::DeltaExtended{T}) where {f, T} = begin
    μ_in, Σ_in   = mean_cov(m_in)
    (A, b) = localLinearizationSingleIn(f, m_in)
    m = A*μ_in + b
    V = A*Σ_in*A'
    return NormalMeanPrecision(m, V)
end