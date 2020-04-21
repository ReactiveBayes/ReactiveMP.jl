export AbstractVariable, forward_message, backward_message
export inference

using Rocket

abstract type AbstractVariable end

inference(variable) = combineLatest((forward_message(variable), backward_message(variable)), true, (AbstractMessage, multiply))

forward_message(variable::V)  where V = error("You probably forgot to implement forward_message(variable::$V)")
backward_message(variable::V) where V = error("You probably forgot to implement backward_message(variable::$V)")
