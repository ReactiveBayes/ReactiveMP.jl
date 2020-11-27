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

conjugate_type(::Type{ <: MvNormalMeanCovariance }, ::Type{ Val{ :out } }) = MvNormalMeanCovariance
conjugate_type(::Type{ <: MvNormalMeanCovariance }, ::Type{ Val{ :μ } })   = MvNormalMeanCovariance
conjugate_type(::Type{ <: MvNormalMeanCovariance }, ::Type{ Val{ :Σ } })   = InverseWishart