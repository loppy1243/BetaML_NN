export Model, loss, hyperparams, optimizer, @model, predict, ModelOutput

import Flux
using Flux.Tracker: data

abstract type ModelOutput end

struct Model{Out<:ModelOutput, M, L<:Function, O<:Function} <: Function
    model::M
    loss::L
    optimizer::O
    params::Vector{Any}
    hyperparams::Dict{Symbol}

    Model{Out, M, L, O}(model, loss, opt, params, pairs...) where
         {Out<:ModelOutput, M, L<:Function, O<:Function} =
        new(model, loss, opt, params, Dict(pairs...))
end
function Model{Out}(model, loss, opt, params, pairs...) where Out<:ModelOutput
    lossf = (xs...) -> loss(model, xs...)
    Model{Out, typeof(model), typeof(lossf), typeof(opt)}(model, lossf, opt, params, pairs...)
end

(m::Model{M, L, O})(xs...) where {M, L<:Function, O<:Function} = m.model(xs...)

loss(m::Model) = m.loss
loss(m::Model, xs...) = m.loss(xs...)

hyperparams(m::Model) = m.hyperparams
Flux.params(m::Model) = m.params

# Can we just set m.optimizer = optimizer(params(m)) ?
optimizer(m::Model) = m.optimizer(Flux.params(m))

predict(model, args...) = throw(MethodError(predict, (model, args...)))

#function train(file, model, xs, ys; lastepoch=1, callback=() -> nothing)
#    println("Training model...")
#
#    batches = Iterators.partition(zip(julienne(xs, 2:3), julienne(ys, 2)), BATCHSIZE) #=
#           =# |> collect
#
#    epoch = lastepoch
#    try while epoch < lastepoch + EPOCHS
#        batch = rand(batches)
#
#        lossval = sum(p -> applyloss(model, p...), batch)
#        println("Epoch ", epoch, ", loss: ", lossval)
#        
#        x, y = rand(batch)
#        callback(x, y)
#
#        Flux.Optimise.train!(loss(model), batch, optimizer(model))
#
#        epoch % 100 == 0 && save(file, model, epoch)
#        epoch += 1
#    end catch ex
#        ex isa InterruptException ? interrupt() : rethrow()
#    finally epoch != 1 && save(file, model, epoch) end
#end

include("model_macro.jl")
import .ModelMacro: @model
