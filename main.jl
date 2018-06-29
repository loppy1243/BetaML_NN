julienne(A, dims) = mapslices(x -> [x], A, dims)
julienne(f, A, dims) = mapslices(x -> [f(x)], A, dims)

catloss(model; ϵ=1, λ=1) = (event, startcell) -> begin
    pred = model(event)

    -log(ϵ + max(0, pred[startcell...])) + abs(λ*(sum(pred) - pred[startcell...]))
end

using Base.Iterators: partition
function train(file, model, loss, xs, ys; lastepoch=1, opt=SGD)
    batches = partition(zip(julienne(xs, (1, 2)), julienne(ys, 1)), BATCHSIZE) |> collect

    epoch = lastepoch
    try while epoch < lastepoch + EPOCHS
        batch = rand(batches)

        lossval = sum(p -> loss(p...) |> data, batch)
        println("Epoch ", epoch, ", loss: ", lossval)
        
        x, y = rand(batch)
        plotevent(x, map(data, model(x)), cellpoint(y), loss(x, y) |> data) |> gui

        train!(loss, batch, opt(params(model)))

        if epoch % 100 == 0
            print("Saving model to \"", file, "\"... ")
            BSON.@save file model epoch
            println(" Done.")
        end

        epoch += 1
    end catch ex
        ex isa InterruptException ? interrupt() : rethrow()
    finally if epoch != 1
        print("Saving model to \"", file, "\"...")
        BSON.@save file model epoch
        println(" Done.")
    end end
end

function cellpoint(cell)
    xy = [cell[2], cell[1]]

    # =(xy - 1 + 1/2)
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
