
# EM in principle can reuse rule from Marginalisation constraint and only change marginal computation strategy in prod function.
# That needs to be revisited later
function rule(fform, on, ::ExpectationMaximisation, messages_names, messages, marginals_names, marginals, meta, __node)
    return rule(fform, on, Marginalisation(), messages_names, messages, marginals_names, marginals, meta, __node)
end