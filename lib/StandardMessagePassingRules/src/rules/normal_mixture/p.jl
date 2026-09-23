# The component's precision likelihood, weighted by its responsibility z: a Gamma for a
# univariate component, and for a d-dimensional one a Wishart with 1 + z + d degrees of freedom
# and the inverse scale z E[(out - m)(out - m)ᵀ].
function mixture_precision_likelihood(::Type{Univariate}, q_out, q_m, z)
    m_mean, m_var = mean_var(q_m)
    out_mean, out_var = mean_var(q_out)
    return GammaShapeRate(one(z) + z / 2, z * (out_var + m_var + abs2(out_mean - m_mean)) / 2)
end

mixture_precision_likelihood(::Type{Multivariate}, q_out, q_m, z) =
    WishartFast(one(z) + z + ndims(q_out), z * difference_moment(q_out, q_m))

@define_message_update_rule(
    node = NormalMixture, target = (:p, k),
    args = (q[:out]::Any, q[:switch]::Any, q[:m][k]::Any),
    body = (args) -> mixture_precision_likelihood(variate_form(typeof(args.q[:out])), args.q[:out], args.q[:m][k], probvec(args.q[:switch])[k]),
)

@define_message_update_rule(
    node = NormalMixture, target = (:p, k),
    args = (q[:out]::Any, q[:switch]::PointMass{<:Real}, q[:m][k]::Any),
    body = (args) -> error(INTEGER_SWITCH),
)
