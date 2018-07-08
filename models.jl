export Model, applymodel, loss, applyloss, hyperparams, optimizer

import Flux
using Flux.Tracker: data

struct Losses <: Function
    funcs::Vector{Function}

    Losses(x::Vector{Function}) = new(x)
    Losses(xs...) = new(collect(Function, xs))
end
(ls::Losses)(xs...) = map(l -> l(xs...), ls)

Base.convert(::Type{Losses}, t::NTuple{N, Function}) where N = Losses(collect(t))
Base.convert(::Type{Losses}, t::AbstractVector{Function}) = Losses(collect(t))
Base.getindex(ls::Losses, i) = ls.funcs[i]

struct Model{M, L<:Function, O<:Function} <: Function
    model::M
    loss::L
    optimizer::O
    params::Vector{Any}
    hyperparams::Dict{Symbol}

    Model{M, L, O}(model, loss, opt, params, pairs...) where {M, L<:Function, O<:Function} =
        new(model, loss, opt, params, Dict(pairs...))
end
function Model(model, lossgen, opt, params, pairs...)
    loss = convert(Losses, map(l -> l(model), lossgen))
    Model{typeof(model), Losses, typeof(opt)}(model, loss, opt, params, pairs...)
end
function Model(model, lossgen::Function, opt, params, pairs...)
    loss = lossgen(model)
    Model{typeof(model), typeof(loss), typeof(opt)}(model, loss, opt, params, pairs...)
end

(m::Model{M, L, O})(xs...) where {M, L<:Function, O<:Function} = m.model(xs...)

loss(m::Model) = m.loss
loss(m::Model, x, y) = m.loss(x, y)

hyperparams(m::Model) = m.hyperparams
Flux.params(m::Model) = m.params

# Can we just set m.optimizer = optimizer(params(m)) ?
optimizer(m::Model) = m.optimizer(Flux.params(m))

function train(file, model, xs, ys; lastepoch=1, callback=() -> nothing)
    println("Training model...")

    batches = Iterators.partition(zip(julienne(xs, 2:3), julienne(ys, 2)), BATCHSIZE) #=
           =# |> collect

    epoch = lastepoch
    try while epoch < lastepoch + EPOCHS
        batch = rand(batches)

        lossval = sum(p -> applyloss(model, p...), batch)
        println("Epoch ", epoch, ", loss: ", lossval)
        
        x, y = rand(batch)
        callback(x, y)

        Flux.Optimise.train!(loss(model), batch, optimizer(model))

        epoch % 100 == 0 && save(file, model, epoch)
        epoch += 1
    end catch ex
        ex isa InterruptException ? interrupt() : rethrow()
    finally epoch != 1 && save(file, model, epoch) end
end
