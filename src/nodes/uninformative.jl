export Uninformative, make_node

struct Uninformative end

@node(
    formtype   => Uninformative,
    sdtype     => Deterministic,
    interfaces => [ out ]
)