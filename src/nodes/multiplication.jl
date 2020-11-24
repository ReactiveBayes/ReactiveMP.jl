export make_node

@node(
    formtype   => typeof(*),
    sdtype     => Deterministic,
    interfaces => [ out, A, in ]
)