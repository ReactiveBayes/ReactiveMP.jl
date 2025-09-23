using BenchmarkTools
using ReactiveMP
import ReactiveMP: Marginal, Contingency, Categorical, Marginalisation, normalize!
using ReactiveMP.BayesBase
using ReactiveMP.ExponentialFamily

function add_discrete_transition_categoricals_rule_benchmarks(SUITE)
    SUITE["Categoricals"] = BenchmarkGroup(["Rules", "DiscreteTransition"])
    for n_categories in 2:5
        for categorical_size in [3, 5, 10, 20]
            discrete_transition_mean_field_categoricals_rule(SUITE["Categoricals"], n_categories, categorical_size)
            discrete_transition_bp_categoricals_rule(SUITE["Categoricals"], n_categories, categorical_size)
            if n_categories > 4
                discrete_transition_structured_categoricals_rule(SUITE["Categoricals"], n_categories, categorical_size)
            end
        end
    end
    discrete_transition_bp_categoricals_rule(SUITE["Categoricals"], 10, 2)
    discrete_transition_bp_categoricals_rule(SUITE["Categoricals"], 20, 2)
end

function discrete_transition_structured_categoricals_rule(SUITE, n_categories, cat_size)
    # Create messages for some interfaces
    n_messages = div(n_categories, 2)  # Use half of the interfaces as messages
    incoming_messages = ntuple(_ -> ReactiveMP.Message(Categorical(normalize!(rand(cat_size), 1)), false, false, nothing), n_messages) # One less message since we're computing :out

    # Create message names
    incoming_messages_name = [:in]
    for i in 2:(n_messages)
        push!(incoming_messages_name, Symbol("T$(i-1)"))
    end
    incoming_messages_name = Tuple(incoming_messages_name)

    # Create marginals for the remaining interfaces including 'a'
    n_marginals = n_categories - n_messages  # +1 for 'a', +1 since we removed :out from messages
    incoming_marginals = ntuple(i -> if i == 1
        ReactiveMP.Marginal(DirichletCollection(rand(ntuple(_ -> cat_size, n_categories)...)), false, false, nothing)
    else
        ReactiveMP.Marginal(Categorical(normalize!(rand(cat_size), 1)), false, false, nothing)
    end, n_marginals)

    # Create marginal names
    incoming_marginals_name = [:a]
    for i in (n_messages):(n_categories - 2)
        push!(incoming_marginals_name, Symbol("t$(i)"))
    end
    incoming_marginals_name = Tuple(incoming_marginals_name)

    # Set up the benchmark
    SUITE["Structured"]["$(n_categories) categories, $(cat_size) size"] = @benchmarkable ReactiveMP.rule(
        DiscreteTransition,
        Val(:out),
        Marginalisation(),
        Val($incoming_messages_name),
        $incoming_messages,
        Val($incoming_marginals_name),
        $incoming_marginals,
        nothing,
        nothing,
        ReactiveMP.call_rule_make_node(DiscreteTransition, DiscreteTransition, nothing)
    )
end

function discrete_transition_mean_field_categoricals_rule(SUITE, n_categories, cat_size)
    incoming_marginals = ntuple(i -> if i == 1
        ReactiveMP.Marginal(DirichletCollection(rand(ntuple(_ -> cat_size, n_categories)...)), false, false, nothing)
    else
        ReactiveMP.Marginal(Categorical(normalize!(rand(cat_size), 1)), false, false, nothing)
    end, n_categories)
    incoming_marginals_name = [:a, :in]
    for i in 1:(n_categories - 2)
        push!(incoming_marginals_name, Symbol("t$i"))
    end
    incoming_marginals_name = Tuple(incoming_marginals_name)
    SUITE["Mean Field"]["$(n_categories) categories, $(cat_size) size"] = @benchmarkable ReactiveMP.rule(
        DiscreteTransition,
        Val(:out),
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

function discrete_transition_bp_categoricals_rule(SUITE, n_categories, cat_size)
    incoming_marginal = (ReactiveMP.Marginal(DirichletCollection(rand(ntuple(_ -> cat_size, n_categories)...)), false, false, nothing),)
    incoming_marginal_name = (:a,)
    incoming_messages = ntuple(_ -> ReactiveMP.Message(Categorical(normalize!(rand(cat_size), 1)), false, false, nothing), n_categories - 1)
    incoming_messages_name = [:in]
    for i in 1:(n_categories - 2)
        push!(incoming_messages_name, Symbol("T$i"))
    end
    incoming_messages_name = Tuple(incoming_messages_name)

    SUITE["Belief Propagation"]["$(n_categories) categories, $(cat_size) size"] = @benchmarkable ReactiveMP.rule(
        DiscreteTransition,
        Val(:out),
        Marginalisation(),
        Val($incoming_messages_name),
        $incoming_messages,
        Val($incoming_marginal_name),
        $incoming_marginal,
        nothing,
        nothing,
        ReactiveMP.call_rule_make_node(DiscreteTransition, DiscreteTransition, nothing)
    )
end
