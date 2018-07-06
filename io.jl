import BSON

using Plots
using BetaML_Data

function save(file, model)
    print("Saving model to \"", file, "\"... ")
    BSON.@save file model
    println(" Done.")
end

function load(file)
    print("Loading model from \"", file, "\"...")
    ret = BSON.load(file)[:model]
    println(" Done.")
    ret
end
try_load(file, model) = isfile(file) ? load(file) : model

function readdata(range)
    print("Reading data from \"", DATAFILE, "\"...")
    events, inits = BetaML_Data.read(DATAFILE, range) .|> x -> convert(Array{Float64}, x)
    println(" Done.")
    (events, inits[:, 2:3])
end

function plotpoint!(plt, p; kws...)
    xmin, xmax = xlims(plt)
    ymin, ymax = ylims(plt)

    xy = @. (p - XYOFF)/(XYMAX-XYMIN)*[xmax-xmin, ymin-ymax] + [xmin, ymax]

    scatter!(plt, [xy[1]], [xy[2]], legend=false; kws...)
end
plotpoint!(p; kws...) = plotpoint!(Plots.current(), p; kws...)

function plotevent(event, pred_grid, point, lossval)
    input_plt = spy(flipdim(event, 1), colorbar=false)
    plotpoint!(point)

    output_plt = spy(flipdim(pred_grid, 1), title="Loss="*string(lossval), colorbar=false)

    plot(layout=(1, 2), input_plt, output_plt, aspect_ratio=1)
end
