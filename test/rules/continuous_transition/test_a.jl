@call_rule ContinuousTransition(:h, Marginalisation) (q_y_x = MvNormalMeanCovariance(randn(5), diageye(5)), q_Λ = Wishart(2, diageye(2)), meta = CTMeta(2, 3))
