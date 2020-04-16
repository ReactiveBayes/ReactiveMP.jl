export AbstractVariable, forward_message, backward_message
export inference

using Rocket

abstract type AbstractVariable end

inference(variable) = combineLatest(forward_message(variable), backward_message(variable), isbatch = true, transformType = AbstractMessage, transformFn = t -> multiply(t[1], t[2]))

forward_message(variable::V)  where V = error("You probably forgot to implement forward_message(variable::$V)")
backward_message(variable::V) where V = error("You probably forgot to implement backward_message(variable::$V)")
