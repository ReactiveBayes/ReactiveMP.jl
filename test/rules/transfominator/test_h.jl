@call_rule Transfominator(:h, Marginalisation) (q_y_x = MvNormalMeanCovariance(randn(5), diageye(5)), q_Λ = Wishart(2, diageye(2)), meta = TMeta(2, 3))
