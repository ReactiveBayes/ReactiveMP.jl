@call_rule Transfominator(:x, Marginalisation) (
    m_y = MvNormalMeanPrecision(randn(2), diageye(2)), q_h = MvNormalMeanPrecision(randn(6), diageye(6)), q_Λ = Wishart(2, diageye(2)), meta = TMeta(2, 3)
)
