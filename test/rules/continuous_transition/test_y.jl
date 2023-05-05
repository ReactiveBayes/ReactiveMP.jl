@call_rule ContinuousTransition(:y, Marginalisation) (
    m_x = MvNormalMeanPrecision(randn(3), diageye(3)), q_h = MvNormalMeanPrecision(randn(6), diageye(6)), q_Λ = Wishart(2, diageye(2)), meta = CTMeta(2, 3)
)
