export LazyObservable

import Rx
import Rx: Subscribable, on_subscribe!, ValidSubscribable, AbstractActor

mutable struct LazyObservable{D} <: Subscribable{D}
    name :: String
    observable

    LazyObservable{D}(name::String) where D = begin
        self = new()

        self.name = name

        return self
    end
end

define!(lazy::LazyObservable{D}, observable::O) where O where D = define!(as_subscribable(O), D, lazy, observable)

function define!(::ValidSubscribable{S}, ::Type{D}, lazy::LazyObservable{D}, observable) where { S <: D } where D
    lazy.observable = observable
end

function Rx.on_subscribe!(lazy::LazyObservable{D}, actor::A) where { A <: AbstractActor{D} } where D
    if !isdefined(lazy, :observable)
        error("[$(lazy.name)]: Lazy observable is not defined")
    else
        return subscribe!(lazy.observable, actor)
    end
end
