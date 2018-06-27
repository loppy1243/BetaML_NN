module DenseNet

include("readdata.jl")

import BSON
using Flux
using Plots; gr()
using Flux.Optimise: train!
using Flux.Tracker: data

const DATAFILE = "I:\\projects\\temp\\liddick\\BetaScint2DEnergy.txt"
const TRAIN_RANGE = 1 : div(EVENTS, 3)
const VAL_RANGE = div(EVENTS, 3)+1 : 2*div(EVENTS, 3)
const TEST_RANGE = 2*div(EVENTS, 3)+1 : EVENTS
const BATCHES = 1000
const MAX_E = 3060

function _train(file, model, loss, xs, ys; epoch=1, opt=SGD)
    batches = Iterators.partition(zip(xs, ys), BATCHES) |> collect
    ixs_init = IntSet(indices(batches, 1))
    ixs = copy(ixs_init)

    for i = epoch:epoch+BATCHES
        if isempty(ixs); ixs = copy(ixs_init) end

        k = rand(ixs |> collect)
        batch = batches[k]
        setdiff!(ixs, k)

        lossval = sum(p -> loss(p...) |> data, batch)
        println("Epoch ", i, ", loss: ", lossval)
        
        x, y = rand(batch)
        plotevent(x, map(data, model(x)), cellpoint(y), loss(x, y) |> data) |> gui

        train!(loss, batch, opt(params(model)))
        BSON.@save file model epoch=i
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
    input_plt = spy(flipdim(event, 1))
    plotpoint!(point)

    output_plt = spy(flipdim(pred_grid, 1), title="Loss="*string(lossval))

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

const MODELFILE = "catmodel.bson"
include("catdnn.jl")
#include("regdnn.jl")

main() = catmodel_train_main()

end # module DenseNet
