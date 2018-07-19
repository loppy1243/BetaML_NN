using Flux.Tracker: istracked, data

abstract type ModelType

struct Model{T<:ModelType}
    params::Vector{Any}
    hyperparams::Dict{String}
end

predict(m::Model) = (args...,) -> m(args...)
function predict(m::Model, args...)
    x = m(args...)
    istracked(x) ? data(x) : map(data, x)
end

loss(m::Model) = (args...,) -> loss(args...)
