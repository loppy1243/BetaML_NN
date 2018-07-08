module OtherNN

import NNlib

using Flux
using Plots
using BetaML_Data
using ..BetaML_NN

using Flux: throttle, Tracker.data

distloss(ϵ, λ) = model -> (events, points) -> begin
    pred_dists = model(events, Val{:dist})
    cells = mapslices(pointcell, points, 1)

    sum(1:size(cells, 2)) do i
        pred_dist = pred_dists[:, :, i]
        cell_prob = pred_dist[cells[:, i]...]

        -log(ϵ + max(0, cell_prob)) + abs(λ*(sum(pred_dist) - cell_prob))
    end
end

distloss2(ϵ) = model -> (events, points) -> begin
    pred_dists = model(events, Val{:dist})
    cells = mapslice(pointcell, points, 1)

    sum(1:size(cells, 2)) do i
        pred_dist = pred_dists[:, :, i]
        cell_prob = pred_dist[cells[:, i]...]

        -log(ϵ + max(0, cell_prob)) + log(1+ϵ - min(1, cell_prob)) #=
     =# -sum(x -> log(1+ϵ - min(1, x)), pred_dist)
    end
end

regloss(model) = (events, points) -> begin
    pred_dists, pred_rel_points = model(events)
    bare_pred_dists = data(pred_dists)

    sum(1:size(cells, 2)) do i
        point = points[:, i]
        pred_dist = bare_pred_dists[:, :, i]
        pred_cell = ind2sub(pred_dist, indmax(pred_dist)) |> collect
        pred_rel_point = pred_rel_points[pred_cell..., :, i]

        (point - (pred_rel_point + cellpoint(pred_cell))).^2 |> sum
    end
end

struct ConvUnbiased{N, A<:AbstractArray{Float64, 4}, F<:Function}
    activ::F
    weights::A
    stride::NTuple{N, Int}
    pad::NTuple{N, Int}
end
Flux.treelike(ConvUnbiased)
ConvUnbiased(dims::NTuple{N}, chs, activ=identity; stride=map(_->1, dims),
             pad=map(_->1, dims), init=rand) where N =
    ConvUnbiased(activ, param(init(dims..., chs...)), stride, pad)
(c::ConvUnbiased)(x) = c.activ.(NNlib.conv(x, c.weights, stride=c.stride, pad=c.pad))

function other(activ; ϵ=1, λ=1, η=0.1, N=50)
    distchain = Chain(Conv((3, 3), 1=>1, pad=(1, 1)),
                      x -> reshape(x, CELLS, :),
                      softmax,
                      x -> reshape(x, GRIDSIZE..., :))
    pointchain = Chain(Conv((5, 5), 1=>N, first(activ), pad=(2, 2), init=zeros),
                       ConvUnbiased((5, 5), N=>2, pad=(2, 2), init=zeros))

    regularize(x) = reshape(x, GRIDSIZE..., 1, :)
    model(x) = x |> Chain(regularize, y -> (distchain(y), pointchain(y)))
    model(x, ::Type{Val{:dist}}) = x |> Chain(regularize, distchain)
    model(x, ::Type{Val{:point}}) = x |> Chain(reguarize, pointchain)

    Model(model, (distloss(ϵ, λ), regloss), x -> SGD(x, η),
          [params(distchain); params(pointchain)],
          :activ => last(activ), :opt => "SGD", :ϵ => ϵ, :λ => λ, :η => η, :N => N)
end

function batch(xs, sz)
    lastdim = ndims(xs)
    ret = []
    for i in 1:div(size(xs, lastdim), sz)
        push!(ret, slicedim(xs, lastdim, 1+(i-1)*sz:min(size(xs, lastdim), i*sz)))
    end

    ret
end

model1() = other(relu=>"relu"; ϵ=0.1, λ=1, η=0.01, N=50)

function dist_relay_info(batchnum, model, events, points)
    i = rand(1:size(events, 3))
    pred_dist = model(events[:, :, [i]])[1] |> data |> x -> squeeze(x, 3)

    println("$(round(batchnum, 2)):, Event $i, Loss = ",
            loss(model)[1](events[:, :, [i]], points[:, [i]]) |> data)
    println("Conv: ", params(model)[1])
    println("Bias: ", params(model)[2])

    lplt = spy(flipdim(events[:, :, i], 1), colorbar=false, title="Event")
    BetaML_NN.plotpoint!(points[:, i], color=:green)
    rplt = spy(flipdim(pred_dist, 1), colorbar=false, title="Pred. dist.")
    BetaML_NN.plotpoint!(points[:, i], color=:green)

    plot(layout=(1, 2), lplt, rplt, aspect_ratio=1)
    gui()
end

function reg_relay_info(batchnum, model, events, points)
  i = rand(1:size(events, 3))
  pred = model(events[:, :, [i]]) .|> data
  cell_point = cellpoint(pointcell(points[:, i]))
  pred_dist = pred[1] |> x -> squeeze(x, 3)
  pred_cell = ind2sub(pred_dist, indmax(pred_dist)) |> collect
  pred_cell_point = cellpoint(pred_cell)
  pred_point = pred_cell_point + pred[2][pred_cell..., :, 1]

  println("$(round(batchnum, 2)): Event $i, Loss = ",
          loss(model)[2](events[:, :, [i]], points[:, [i]]) |> data)

  lplt = spy(flipdim(events[:, :, i], 1), colorbar=false, title="Event")
  BetaML_NN.plotpoint!(points[:, i], color=:green)
  rplt = spy(flipdim(pred_dist, 1), colorbar=false, title="Pred. dist.")
  BetaML_NN.plotpoint!(pred_cell_point, color=:purple)
  BetaML_NN.plotpoint!(points[:, i], color=:green)
  BetaML_NN.plotpoint!(pred_point, color=:red)

  plot(layout=(1, 2), lplt, rplt, aspect_ratio=1)
  gui()
end

function train(file, model, events, points)
    batches = zip(batch(events, BATCHSIZE), batch(points, BATCHSIZE)) |> collect
    numbatches = length(batches)

    batchnum = Ref(0)
    update_info() = batchnum.x += 1
    t_save = throttle(10) do
        BetaML_NN.save(file, model)
    end
    t_dist_relay_info = throttle(2) do
        dist_relay_info(batchnum.x/numbatches, model, events, points)
    end
    t_reg_relay_info = throttle(2) do
        reg_relay_info(batchnum.x/numbatches, model, events, points)
    end

    println("Training dist. epoch...")
    shuffle!(batches)
    try
        Flux.Optimise.train!(loss(model)[1], batches, optimizer(model),
                             cb=[update_info, t_save, t_dist_relay_info])
    catch ex
        ex isa InterruptException ? interrupt() : rethrow()
    finally
        BetaML_NN.save(file, model)
    end
  
    batchnum.x = 0
    println("Training reg. epoch...")
    shuffle!(batches)
    try
        Flux.Optimise.train!(loss(model)[2], batches, optimizer(model),
                             cb=[update_info, t_save, t_reg_relay_info])
    catch ex
        ex isa InterruptException ? interrupt() : rethrow()
    finally
        BetaML_NN.save(file, model)
    end
end

end # module OtherNN
