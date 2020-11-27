@rule(
    formtype    => Uninformative,
    on          => :out,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return missing
    end
)