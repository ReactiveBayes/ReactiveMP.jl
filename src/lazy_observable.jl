export LazyObservable

import Rx
import Rx: Subscribable, on_subscribe!, ValidSubscribable, AbstractActor

mutable struct LazyObservable{D} <: Subscribable{D}
    name       :: String
    observable

    LazyObservable{D}(name::String) where D = begin
        self = new()

        self.name = name

        return self
    end
end

define!(lazy::LazyObservable{D}, observable::O) where O where D = define!(as_subscribable(O), lazy, observable)

function define!(::ValidSubscribable{S}, lazy::LazyObservable{D}, observable) where { S <: D } where D
    lazy.observable = observable
end

function define!(::ValidSubscribable{S}, lazy::LazyObservable{D}, observable) where S where D
    error("define! failed")
end

function Rx.on_subscribe!(lazy::LazyObservable{D}, actor::A) where { A <: AbstractActor{D} } where D
    return subscribe!(lazy.observable, actor)
end
