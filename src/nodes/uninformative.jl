export make_node, Loose

struct Uninformative end

@node(
    formtype   => Uninformative,
    sdtype     => Deterministic,
    interfaces => [ out ]
)