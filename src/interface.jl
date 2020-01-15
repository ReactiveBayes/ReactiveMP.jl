using Rx

export AbstractInterface, InterfaceIn, InterfaceOut

abstract type AbstractInterface end

define_joint!(interface::I, observable) where { I <: AbstractInterface }       = set!(interface.lazy_jm, observable)
define_sum_product!(interface::I, observable) where { I <: AbstractInterface } = set!(interface.lazy_sp, observable)

sum_product(interface::I) where { I <: AbstractInterface } = interface.sum_product_message

struct InterfaceIn{S, J} <: AbstractInterface
    name :: String

    lazy_sp :: LazyObservable{S}
    lazy_jm :: LazyObservable{J}

    sum_product_message
    joint_message

    InterfaceIn{S, J}(name::String) where { S <: AbstractMessage } where { J <: AbstractMessage } = begin
        lazy_sp = LazyObservable{S}()
        lazy_jm = LazyObservable{J}()

        sum_product_message = lazy_sp# |> replay(1) #|> take(1)
        joint_message       = lazy_jm #|> take(1)

        return new(name, lazy_sp, lazy_jm, sum_product_message, joint_message)
    end
end

struct InterfaceOut{S, J} <: AbstractInterface
    name :: String

    lazy_sp :: LazyObservable{S}
    lazy_jm :: LazyObservable{J}

    sum_product_message
    joint_message

    InterfaceOut{S, J}(name::String) where { S <: AbstractMessage } where { J <: AbstractMessage } = begin
        lazy_sp = LazyObservable{S}()
        lazy_jm = LazyObservable{J}()

        sum_product_message = lazy_sp# |> replay(1) #|> take(1)
        joint_message       = lazy_jm #|> take(1)

        return new(name, lazy_sp, lazy_jm, sum_product_message, joint_message)
    end
end
