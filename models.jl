export Model, applymodel, loss, applyloss, hyperparams, optimizer

import Flux
using Flux.Tracker: data

struct Model{M, L<:Function, O<:Function} <: Function
    model::M
    loss::L
    optimizer::O
    params::Dict{Symbol}

    Model{M, L, O}(model, loss, opt, pairs...) where {M, L<:Function, O<:Function} =
        new(model, loss, opt, Dict(pairs...))
end
function Model(model, lossgen, opt, pairs...)
    loss = lossgen(model)
    Model{typeof(model), typeof(loss), typeof(opt)}(model, loss, opt, pairs...)
end

(m::Model{M, L, O})(x) where {M, L<:Function, O<:Function} = m.model(x)

loss(m::Model) = m.loss
loss(m::Model, x, y) = m.loss(x, y)
applyloss(m::Model) = (x, y) -> m.loss(x, y) |> data
applyloss(m::Model, x, y) = m.loss(x, y) |> data

hyperparams(m::Model) = m.params
Flux.params(m::Model) = Flux.params(m.model)

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
