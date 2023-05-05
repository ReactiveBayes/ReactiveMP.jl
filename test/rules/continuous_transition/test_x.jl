@call_rule ContinuousTransition(:x, Marginalisation) (
    m_y = MvNormalMeanPrecision(randn(2), diageye(2)), q_h = MvNormalMeanPrecision(randn(6), diageye(6)), q_Λ = Wishart(2, diageye(2)), meta = CTMeta(2, 3)
)
