module DenseNet

include("readdata.jl")

import BSON
using Flux
using Plots; gr()
using Flux.Optimise: train!
using Flux.Tracker: data

const DATAFILE = "I:\\projects\\temp\\liddick\\/BetaScint2DEnergy.txt"
const TRAIN_RANGE = 1 : div(EVENTS, 3)
const VAL_RANGE = div(EVENTS, 3)+1 : 2*div(EVENTS, 3)
const TEST_RANGE = 2*div(EVENTS, 3)+1 : EVENTS
const BATCHES = 1000
const MAX_E = 3060

catmodel(n, activ) =
    Chain(x -> x[:]/MAX_E, Dense(CELLS, n, activ), Dense(n, CELLS), softmax,
          x -> reshape(x, GRIDSIZE...))
regmodel(n, activ) = Chain(Dense(CELLS, n, activ), Dense(n, 2))

catloss(model, ϵ=1, scale=1) = (event, startcell) -> begin
    pred = model(event)

    -log(ϵ + pred[startcell...]) + scale*(sum(pred) - pred[startcell...])
end
regloss(model) = (event, startpos) -> norm(startpos - model(event))

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

const MODELFILE = "catmodel.bson"

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

function catmodel_train_main()
    print("Reading data from \"", DATAFILE, "\"...")
    events, inits = readdata(DATAFILE, TRAIN_RANGE)
    cells = map(x -> pointcell(x[2:3]), inits)
    println(" Done.")

    if isfile(MODELFILE)
        print("Loading model from \"", MODELFILE, "\"...")
        BSON.@load MODELFILE model epoch
        println(" Done.")
    else
        model, epoch = catmodel(2CELLS, relu), 1

        print("Initializing model parameters...")
        for param in params(model)
            Flux.Tracker.data(param) .= 0
        end
        println(" Done.")
    end

    println("Training model...")
    train(MODELFILE, model, catloss(model), events, cells, epoch=epoch)
end

createpath(path) = if !ispath(path); mkpath(path) end

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

function catmodel_train_plot()
    print("Reading data from \"", DATAFILE, "\"...")
    events, inits = readdata(DATAFILE, TRAIN_RANGE)
    println(" Done.")

    points = map(x -> x[2:3], inits)

    BSON.@load MODELFILE model

    plotmodel("plots/test/", model, catloss(model), events, points)
end

function catmodel_validate_plot()
    print("Reading data from \"", DATAFILE, "\"...")
    events, inits = readdata(DATAFILE, VALID_RANGE)
    println(" Done.")

    points = map(x -> x[2:3], inits)

    BSON.@load MODELFILE model

    plotmodel("plots/validate/", model, catloss(model), events, points)
end

main() = catmodel_train_main()

end # module DenseNet
