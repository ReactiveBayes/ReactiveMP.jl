using BenchmarkTools
using ReactiveMP
import ReactiveMP: Marginal, Contingency, Categorical, Marginalisation, normalize!
using ReactiveMP.BayesBase

function add_discrete_transition_a_rule_benchmarks(SUITE)
    SUITE["a"] = BenchmarkGroup()
    for n_categories in 2:5
        for categorical_size in [3, 5, 10, 20]
            discrete_transition_mean_field_a_rule(SUITE["a"], n_categories, categorical_size)
            discrete_transition_bp_a_rule(SUITE["a"], n_categories, categorical_size)
        end
    end
end

function discrete_transition_mean_field_a_rule(SUITE, n_categories, cat_size)
    incoming_marginals = ntuple(_ -> ReactiveMP.Marginal(Categorical(normalize!(rand(cat_size), 1)), false, false, nothing), n_categories)
    incoming_marginals_name = [:out, :in]
    for i in 1:(n_categories - 2)
        push!(incoming_marginals_name, Symbol("t$i"))
    end
    incoming_marginals_name = Tuple(incoming_marginals_name)
    SUITE["Mean Field"]["$(n_categories) categories, $(cat_size) size"] = @benchmarkable ReactiveMP.rule(
        DiscreteTransition,
        Val(:a),
        Marginalisation(),
        nothing,
        nothing,
        Val($incoming_marginals_name),
        $incoming_marginals,
        nothing,
        nothing,
        ReactiveMP.call_rule_make_node(DiscreteTransition, DiscreteTransition, nothing)
    )
end

function discrete_transition_bp_a_rule(SUITE, n_categories, cat_size)
    incoming_marginal = (ReactiveMP.Marginal(Contingency(rand(ntuple(_ -> cat_size, n_categories)...)), false, false, nothing),)
    incoming_marginal_name = "out_in_t"
    incoming_marginal_name *= join(1:(n_categories - 2), "_t")
    incoming_marginal_name = (Symbol(incoming_marginal_name),)
    SUITE["Belief Propagation"]["$(n_categories) categories, $(cat_size) size"] = @benchmarkable ReactiveMP.rule(
        DiscreteTransition,
        Val(:a),
        Marginalisation(),
        nothing,
        nothing,
        Val($incoming_marginal_name),
        $incoming_marginal,
        nothing,
        nothing,
        ReactiveMP.call_rule_make_node(DiscreteTransition, DiscreteTransition, nothing)
    )
end
