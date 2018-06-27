module DenseNet

include("readdata.jl")

import BSON
using Flux
using Plots; gr()
using Flux.Optimise: train!
using Flux.Tracker: data

include("consts.jl")

catloss(model; ϵ=1, λ=1) = (event, startcell) -> begin
    pred = model(event)

    -log(ϵ + max(0, pred[startcell...])) + abs(λ*(sum(pred) - pred[startcell...]))
end

function _train(file, model, loss, xs, ys; epoch=1, opt=SGD)
    batches = Iterators.partition(zip(xs, ys), BATCHES) |> collect

    i = epoch
    try
        while i < epoch + BATCHES
            batch = rand(batches)

            lossval = sum(p -> loss(p...) |> data, batch)
            println("Epoch ", i, ", loss: ", lossval)
            
            x, y = rand(batch)
            plotevent(x, map(data, model(x)), cellpoint(y), loss(x, y) |> data) |> gui

            train!(loss, batch, opt(params(model)))

            if i % 100 == 0
                print("Saving model to \"", file, "\"... ")
                BSON.@save file model epoch=i
                println(" Done.")
            end

            i += 1
        end
    finally
        if i != 1
            print("Saving model to \"", file, "\"...")
            BSON.@save file model epoch=i
            println(" Done.")
        end
    end
end
train(file, model, loss, xs, ys; epoch=1, opt=SGD) = try
    _train(file, model, loss, xs, ys; epoch=epoch, opt=opt)
catch ex
    ex isa InterruptException ? interrupt() : rethrow()
end

function cellpoint(cell)
    xy = [cell[2], cell[1]]

    @. (xy - 1/2)/GRIDSIZE*(XYMAX-XYMIN) + XYOFF
end

function pointcell(p)
    fix(x) = iszero(x) ? oneunit(x) : x
    swap(v) = [v[2], v[1]]

    (@. (p - XYOFF)/(XYMAX-XYMIN)*GRIDSIZE |> ceil |> Int |> fix) |> swap
end

function plotpoint!(plt, p)
    xmin, xmax = xlims(plt)
    ymin, ymax = ylims(plt)

    xy = @. (p - XYOFF)/(XYMAX-XYMIN)*[xmax-xmin, ymax-ymin] + [xmin, ymin]

    scatter!(plt, [xy[1]], [xy[2]], legend=false)
end
plotpoint!(p) = plotpoint!(Plots.current(), p)

function plotevent(event, pred_grid, point, lossval)
    input_plt = spy(flipdim(event, 1), colorbar=false)
    plotpoint!(point)

    output_plt = spy(flipdim(pred_grid, 1), title="Loss="*string(lossval), colorbar=false)

    plot(layout=(1, 2), input_plt, output_plt, aspect_ratio=1)
end

function plotmodel(dir, model, loss, events, points)
    digs = ndigits(length(events))
    for (i, (event, point)) in zip(events, points) |> enumerate
        print("Generating plot ", i, "...")

        plotevent(event, map(data, model(event)), point,
                  loss(event, pointcell(point)) |> data)

        println(" Done.")
    end
end

createpath(path) = if !ispath(path); mkpath(path) end

const MODELFILE = "cat_pure_cnnmodel.bson"
#include("catdnn.jl")
#include("regdnn.jl")
include("catcnn.jl")

main() = catcnnmodel_train_main()
end # module DenseNet
