export make_node

@node(
    formtype   => MvNormalMeanCovariance,
    sdtype     => Stochastic,
    interfaces => [ 
        out, 
        (μ, aliases = [ mean ]), 
        (Σ, aliases = [ cov ]) 
    ]
)