module OtherNN

import NNlib

using Flux
using Plots
using BetaML_Data
using ..BetaML_NN

using Flux: throttle, Tracker.data

distloss(ϵ, λ) = (model, events, points) -> distloss(model, events, points, ϵ, λ)
function distloss(model, events, points, ϵ, λ)
    pred_dists = model(events, Val{:dist})
    cells = mapslices(pointcell, points, 1)

    sum(1:size(cells, 2)) do i
        pred_dist = pred_dists[:, :, i]
        cell_prob = pred_dist[cells[:, i]...]

        -log(ϵ + max(0, cell_prob)) + abs(λ*(sum(pred_dist) - cell_prob))
    end
end

distloss2(ϵ) = (model, events, points) -> distloss2(model, events, points, ϵ)
function distloss2(model, events, points, ϵ)
    pred_dists = model(events, Val{:dist})
    cells = mapslices(pointcell, points, 1)

    sum(1:size(cells, 2)) do i
        pred_dist = pred_dists[:, :, i]
        cell_prob = pred_dist[cells[:, i]...]

        -log(ϵ + max(0, cell_prob)) + log(1+ϵ - min(1, cell_prob)) #=
     =# -sum(x -> log(1+ϵ - min(1, x)), pred_dist)
    end
end

function regloss(model, events, points)
    pred_dists, pred_rel_points = model(events)
    bare_pred_dists = data(pred_dists)

    sum(1:size(points, 2)) do i
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

@model function other(activ=>activ_name, ϵ, λ, η, N)
    regularize(x) = reshape(x/MAX_E, GRIDSIZE..., 1, :)
    distchain = Chain(Conv((3, 3), 1=>1, pad=(1, 1), init=zeros),
                      x -> reshape(x, CELLS, :),
                      softmax,
                      x -> reshape(x, GRIDSIZE..., :))
    pointchain = Chain(Conv((5, 5), 1=>N, activ, pad=(2, 2), init=zeros),
                       ConvUnbiased((5, 5), N=>2, pad=(2, 2), init=zeros))

    @params [params(distchain); params(pointchain)]
    @opt x -> SGD(x, η)
    @loss(model, x, y) do
        ()                  -> fcat(distloss(ϵ, λ), regloss)(model, x, y)
        ::Type{Val{:dist}}  -> distloss(model, x, y, ϵ, λ)
        ::Type{Val{:point}} -> regloss(model, x, y)
    end
    @model(x) do
        ()                  -> regularize(x) |> fcat(distchain, pointchain)
        ::Type{Val{:dist}}  -> regularize(x) |> distchain
        ::Type{Val{:point}} -> regularize(x) |> pointchain
    end

#    function pointpred(x, dists)
#        ret = Array{eltype(x)}(2, size(x, 3))
#        for i in indices(dists, 3)
#            dist = pred_dists[:, :, i]
#            cell = ind2sub(dist, indmax(dist))
#            cell_point = cellpoint(cell)
#            rel_point = rel_point_dists[pred_cell..., :, i]
#            ret[:, i] .= cell_point + rel_point
#        end
#
#        ret
#    end
#
#    @model(x) do
#        () -> begin
#            y = regularize(x)
#            dist = distchain(y)
#            (dist, pointpred(y, data(dist)))
#        end
#        ::Type{Val{:dist}} -> regularize(x) |> distchain
#        ::Type{Val{:point}} -> begin
#            y = regularize(x)
#            pointpred(y, distchain(y) |> data)
#        end
#    end
end

@model function other2(activ=>activ_name, ϵ, η, N)
    regularize(x) = reshape(x/MAX_E, GRIDSIZE..., 1, :)
    distchain = Chain(Conv((3, 3), 1=>1, pad=(1, 1), init=zeros),
                      x -> reshape(x, CELLS, :),
                      softmax,
                      x -> reshape(x, GRIDSIZE..., :))
    pointchain = Chain(Conv((5, 5), 1=>N, activ, pad=(2, 2), init=zeros),
                       ConvUnbiased((5, 5), N=>2, pad=(2, 2), init=zeros))

    @params [params(distchain); params(pointchain)]
    @opt x -> SGD(x, η)
    @loss(model, x, y) do
        ()                  -> fcat(distloss2(ϵ), regloss)(model, x, y)
        ::Type{Val{:dist}}  -> distloss2(model, x, y, ϵ)
        ::Type{Val{:point}} -> regloss(model, x, y)
    end
    @model(x) do
        ()                  -> regularize(x) |> fcat(distchain, pointchain)
        ::Type{Val{:dist}}  -> regularize(x) |> distchain
        ::Type{Val{:point}} -> regularize(x) |> pointchain
    end
end

function batch(xs, sz)
    lastdim = ndims(xs)
    ret = []
    for i in 1:div(size(xs, lastdim), sz)
        push!(ret, slicedim(xs, lastdim, 1+(i-1)*sz:min(size(xs, lastdim), i*sz)))
    end

    ret
end

model1() = other(relu=>"relu", 0.1, 1, 0.1, 50)
model2() = other2(relu=>"relu", 0.1, 0.1, 50)

function dist_relay_info(batchnum, numbatches, model, events, points)
    i = rand(1:size(events, 3))
    pred_dist = model(events[:, :, [i]], Val{:dist}) |> data |> x -> squeeze(x, 3)

    println("$batchnum/$numbatches:, Event $i, Loss = ",
            loss(model, events[:, :, [i]], points[:, [i]], Val{:dist}) |> data)

    lplt = spy(flipdim(events[:, :, i], 1), colorbar=false, title="Event")
    BetaML_NN.plotpoint!(points[:, i], color=:green)
    rplt = spy(flipdim(pred_dist, 1), colorbar=false, title="Pred. dist.")
    BetaML_NN.plotpoint!(points[:, i], color=:green)

    plot(layout=(1, 2), lplt, rplt, aspect_ratio=1)
    gui()
end

function reg_relay_info(batchnum, numbatches, model, events, points)
  i = rand(1:size(events, 3))
  pred = model(events[:, :, [i]]) .|> data
  cell_point = cellpoint(pointcell(points[:, i]))
  pred_dist = pred[1] |> x -> squeeze(x, 3)
  pred_cell = ind2sub(pred_dist, indmax(pred_dist)) |> collect
  pred_cell_point = cellpoint(pred_cell)
  pred_point = pred_cell_point + pred[2][pred_cell..., :, 1]

  println("$batchnum/$numbatches: Event $i, Loss = ",
          loss(model, events[:, :, [i]], points[:, [i]], Val{:point}) |> data)

  lplt = spy(flipdim(events[:, :, i], 1), colorbar=false, title="Event")
  BetaML_NN.plotpoint!(points[:, i], color=:green)
  rplt = spy(flipdim(pred_dist, 1), colorbar=false, title="Pred. dist.")
  BetaML_NN.plotpoint!(pred_cell_point, color=:purple)
  BetaML_NN.plotpoint!(points[:, i], color=:green)
  BetaML_NN.plotpoint!(pred_point, color=:red)

  plot(layout=(1, 2), lplt, rplt, aspect_ratio=1)
  gui()
end

macro try_and_save(pair, expr)
    model = esc(pair.args[2])
    file = esc(pair.args[3])
    quote
        try
            $(esc(expr))
        catch ex
            ex isa InterruptException ? interrupt() : rethrow()
        finally
            BetaML_NN.save($file, $model)
        end
    end
end

function train(file, model, events, points; load=true, train_dist=true, train_reg=true)
    model = isfile(file) && load ? BetaML_NN.load(file) : model

    batches = zip(batch(events, BATCHSIZE), batch(points, BATCHSIZE)) |> collect
    numbatches = length(batches)

    batchnum = Ref(0)
    update_info() = batchnum.x += 1
    t_save = throttle(20) do
        BetaML_NN.save(file, model)
    end
    t_dist_relay_info = throttle(3) do
        dist_relay_info(batchnum.x, numbatches, model, events, points)
    end
    t_reg_relay_info = throttle(3) do
        reg_relay_info(batchnum.x, numbatches, model, events, points)
    end

    train_dist && @try_and_save model=>file begin
        println("Training dist. epoch...")
        shuffle!(batches)
        Flux.Optimise.train!((x, y) -> loss(model, x, y, Val{:dist}), batches, optimizer(model),
                             cb=[update_info, t_save, t_dist_relay_info])
    end

    train_reg && @try_and_save model=>file begin
        batchnum.x = 0
        println("Training reg. epoch...")
        shuffle!(batches)
        Flux.Optimise.train!((x, y) -> loss(model, x, y, Val{:point}), batches, optimizer(model),
                             cb=[update_info, t_save, t_reg_relay_info])
    end
end

end # module OtherNN
