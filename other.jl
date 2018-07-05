module OtherNN

using Flux
using Plots
using BetaML_Data
using ..BetaML_NN

using Flux: throttle, Tracker.data

distloss(ϵ, λ) = model -> (events, points) -> begin
    pred_dists = model(events)[1]
    cells = mapslices(pointcell, points, 1)

    sum(1:size(cells, 2)) do i
        pred_dist = pred_dists[:, :, i]
        cell_prob = pred_dist[cells[:, i]...]

        -log(ϵ + max(0, cell_prob)) + abs(λ*(sum(pred_dist) - cell_prob))
    end
end

regloss(model) = (events, points) -> begin
    preds = model(events)
    cells = mapslices(pointcell, points, 1)

    sum(1:size(cells, 2)) do i
        point = points[:, i]
        pred_dist = preds[1][:, :, i]
        pred_cell = ind2sub(pred_dist, indmax(pred_dist)) |> collect
        pred_rel_point = preds[2][pred_cell..., :, i]

        norm(point - (pred_rel_point + cellpoint(pred_cell)))
    end
end

function other(activ; ϵ=1, λ=1, η=0.1)
    convlayer = Conv((3, 3), 1=>1, pad=(1, 1))
    pointlayer = Conv((3, 3), 1=>2, pad=(1, 1))

    chain1 = Chain(convlayer,
                   x -> reshape(x, CELLS, :),
                   softmax,
                   x -> reshape(x, GRIDSIZE..., :))
    chain2 = Chain(convlayer,
                   x -> first(activ).(x),
                   pointlayer)

    Model(Chain(x -> reshape(x, GRIDSIZE..., 1, :),
                x -> (chain1(x), chain2(x))),
          (distloss(ϵ, λ), regloss),
          x -> SGD(x, η),
          :activ => last(activ), :opt => "SGD", :ϵ => ϵ, :λ => λ, :η => η)
end

function batch(xs, sz)
    lastdim = ndims(xs)
    ret = []
    for i in 1:div(size(xs, lastdim), sz)
        push!(ret, slicedim(xs, lastdim, 1+(i-1)*sz:min(size(xs, lastdim), i*sz)))
    end

    ret
end

model1() = other(relu=>"relu"; ϵ=0.1, λ=1, η=0.1)

function train(file, model, events, points)
    t_save = throttle(() -> BetaML_NN.save(file, model, 1), 10)

    distcb() = (t_save(); (() -> begin
        i = rand(1:size(events, 3))
        pred_dist = model(events[:, :, [i]])[1] |> data |> x -> squeeze(x, 3)
        cell_point = cellpoint(pointcell(points[:, i]))

        println("Event $i, Loss = ", loss(model)[1](events[:, :, [i]], points[:, [i]]) #=
                                  =# |> data)

        lplt = spy(flipdim(events[:, :, i], 1), colorbar=false, title="Event")
        BetaML_NN.plotpoint!(cell_point, color=:blue)
        rplt = spy(flipdim(pred_dist, 1), colorbar=false, title="Pred. dist.")
        BetaML_NN.plotpoint!(cell_point, color=:blue)

        plot(layout=(1, 2), lplt, rplt, aspect_ratio=1)
        gui()
    end) |> f -> throttle(f, 2) |> f -> f())

    regcb() = (t_save(); (() -> begin
        i = rand(1:size(events, 3))
        pred = model(events[:, :, [i]]) .|> data
        cell_point = cellpoint(pointcell(points[:, i]))
        pred_dist = pred[1] |> x -> squeeze(x, 3)
        pred_cell = ind2sub(pred_dist, indmax(pred_dist)) |> collect
        pred_point = pointcell(pred_cell) + pred[2][pred_cell..., :, 1]

        println("Event $i, Loss = ", loss(model)[2](events[:, :, [i]], points[:, [i]]) #=
                                  =# |> data)

        lplt = spy(flipdim(events[:, :, i], 1), colorbar=false, title="Event")
        BetaML_NN.plotpoint!(cell_point, color=:blue)
        BetaML_NN.plotpoint!(points[:, i], color=:green)
        rplt = spy(flipdim(pred_dist, 1), colorbar=false, title="Pred. dist.")
        BetaML_NN.plotpoint!(cell_point, color=:blue)
        BetaML_NN.plotpoint!(pred_cell, color=:purple)
        BetaML_NN.plotpoint!(points[:, i], color=:green)
        BetaML_NN.plotpoint!(pred_point, color=:red)

        plot(layout=(1, 2), lplt, rplt, aspect_ratio=1)
        gui()
    end) |> f -> throttle(f, 2) |> f -> f())

    batches = zip(batch(events, BATCHSIZE), batch(points, BATCHSIZE)) |> collect

    println("Training dist. epoch...")
    shuffle!(batches)
    Flux.Optimise.train!(loss(model)[1], batches, optimizer(model),
                         cb=distcb)
   
    println("Training reg. epoch...")
    shuffle!(batches)
    Flux.Optimise.train!(loss(model)[2], batches, optimizer(model),
                         cb=throttle(regcb, 2))
    
    BetaML_NN.save(file, model, 1)
end

end # module OtherNN
