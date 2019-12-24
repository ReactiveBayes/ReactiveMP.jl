using Rx

export AbstractInterface, InterfaceIn, InterfaceOut

abstract type AbstractInterface end

define_joint!(interface::I, observable) where { I <: AbstractInterface }       = define!(interface.joint_message, observable)
define_sum_product!(interface::I, observable) where { I <: AbstractInterface } = define!(interface.sum_product_message, observable)

sum_product(interface::I) where { I <: AbstractInterface } = interface.sum_product_message

struct InterfaceIn{S, J} <: AbstractInterface
    name :: String

    sum_product_message :: LazyObservable{S}
    joint_message       :: LazyObservable{J}

    InterfaceIn{S, J}(name::String) where { S <: AbstractMessage } where { J <: AbstractMessage } = begin
        sum_product_message = LazyObservable{S}("[$name]: sumProductMessage")
        joint_message       = LazyObservable{J}("[$name]: jointMessage")

        return new(name, sum_product_message, joint_message)
    end
end

struct InterfaceOut{S, J} <: AbstractInterface
    name :: String

    sum_product_message :: LazyObservable{S}
    joint_message       :: LazyObservable{J}

    InterfaceOut{S, J}(name::String) where { S <: AbstractMessage } where { J <: AbstractMessage } = begin
        sum_product_message = LazyObservable{S}("[$name]: sumProductMessage")
        joint_message       = LazyObservable{J}("[$name]: jointMessage")

        return new(name, sum_product_message, joint_message)
    end
end
