# The joint of the inputs. With one of them known, the other's marginal is its message times its
# prior: in1 = out + in2, and in2 = in1 - out.
# With both Gaussian, a joint Gaussian over [in1; in2].

@define_marginal_update_rule(
    node = -, target = (:in1, :in2),
    args = (m[:out]::NormalDistributionsFamily, m[:in1]::NormalDistributionsFamily, m[:in2]::PointMass),
    body = (args) -> promoted_cluster(
        FactorizedCluster((:in1,) => prod(ClosedProd(), sum_message(args.m[:out], args.m[:in2]), args.m[:in1]), (:in2,) => args.m[:in2]),
        args.m[:out], args.m[:in1], args.m[:in2],
    ),
)

@define_marginal_update_rule(
    node = -, target = (:in1, :in2),
    args = (m[:out]::NormalDistributionsFamily, m[:in1]::PointMass, m[:in2]::NormalDistributionsFamily),
    body = (args) -> promoted_cluster(
        FactorizedCluster((:in1,) => args.m[:in1], (:in2,) => prod(ClosedProd(), difference_message(args.m[:in1], args.m[:out]), args.m[:in2])),
        args.m[:out], args.m[:in1], args.m[:in2],
    ),
)

@define_marginal_update_rule(
    node = -, target = (:in1, :in2),
    args = (m[:out]::NormalDistributionsFamily, m[:in1]::NormalDistributionsFamily, m[:in2]::NormalDistributionsFamily),
    body = (args) -> input_joint(args.m[:out], args.m[:in1], args.m[:in2], -1),
)
