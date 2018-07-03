export RegCNN
module RegCNN

import BSON

using Flux
using Plots
using BetaML_Data
using ..BetaML_NN

using StatsBase: mode

lossgen1(ϵ, λ, δ) = model -> (event, pos) -> begin
    cell = pointcell(pos)
    pred_dist, pred_pos = model(event)

    -log(ϵ + max(0, pred_dist[cell...])) + abs(λ*(sum(pred_dist) - pred_dist[cell...])) #=
 =# + δ*norm(pos - pred_pos)
end

lossgen2(ϵ, λ, δ) = model -> (event, pos) -> begin
    cell = pointcell(pos)
    pred_dist, pred_pos = model(event)

    -log(ϵ + max(0, pred_dist[cell...])) + abs(λ*(sum(pred_dist) - pred_dist[cell...])) #=
 =# + δ*log(max(0, norm(pos - pred_pos)))
end

normloss(model) = (event, pos) -> norm(model(event)[2] - pos)

using ..pad
function convdense(activ, lossgen; ϵ=1, λ=1, δ=1, η=0.1)
    convlayergen() = Chain(x -> pad(x/MAX_E),
                           x -> reshape(x, (GRIDSIZE+[2, 2])..., 1, 1),
                           Conv((3, 3), 1=>1),
                           x -> reshape(x, CELLS))
    convlayer1 = convlayergen()
    convlayer2 = convlayergen()
    denselayer = Dense(CELLS, 2)

    cellpred = Chain(convlayer1, softmax, x -> reshape(x, GRIDSIZE...))
    relpospred = Chain(convlayer2, x -> first(activ).(x), denselayer)
    function modelfunc(event)
        pred_dist = cellpred(event)
        cell = (ind2sub(pred_dist, indmax(pred_dist)) |> collect) + [-1, 1] # ???

        (pred_dist, relpospred(event) + cellpoint(cell))
    end

    Model(modelfunc, first(lossgen)(ϵ, λ, δ), x -> SGD(x, η),
          :activ => last(activ), :lossgen => last(lossgen), :ϵ => ϵ, :λ => λ, :δ => δ, :η => η, :opt => "SGD")
end

function staggered(file, activ; η=0.1)
    dist_model = BSON.load(file)[:model]
    denselayer = Dense(CELLS, 2)

    function modelfunc(x)
        pred_dist = dist_model(x) |> Flux.Tracker.data
        cell = (ind2sub(pred_dist, indmax(pred_dist)) |> collect)
        pred_pos = denselayer(reshape(pred_dist, CELLS) .|> first(activ))

        (pred_dist, pred_pos + cellpoint(cell))
    end

    Model(modelfunc, normloss, x -> SGD(x, η),
          :activ => last(activ), :η => η, :opt => "SGD", :loss => "normloss")
end

const model1 = convdense(relu=>"relu", lossgen1=>"lossgen1", ϵ=0.1, λ=1, η=0.1)
const model2 = convdense(relu=>"relu", lossgen2=>"lossgen2", ϵ=0.1, λ=1, η=0.1)
const model3 = staggered("catcnn_onelayer.bson", relu=>"relu"; η=0.1)

function train(modelfile, model, events, points)
    model, epoch = if isfile(modelfile)
        tmp = BetaML_NN.load(modelfile)
        (tmp[:model], tmp[:epoch])
    else
        (model, 1)
    end

    function callback(event, pos)
        cell_pos = cellpoint(pointcell(pos))
        pred_dist, pred_pos = map(Flux.Tracker.data, model(event))

        plt1 = spy(flipdim(event, 1), colorbar=false)

        plt2 = spy(flipdim(pred_dist, 1), colorbar=false)
        BetaML_NN.plotpoint!(pos, color=:green)
        BetaML_NN.plotpoint!(cell_pos, color=:blue)
        BetaML_NN.plotpoint!(pred_pos, color=:red)

        plot(layout=(1, 2), plt1, plt2, aspect_ratio=1)
        gui()
    end

    BetaML_NN.train(modelfile, model, events, points, callback=callback)
end

@modelfunc function disthist(model::Model, events, points; model_name="")
    print("Computing predictions...")
    preds = squeeze(mapslices(x -> model(x)[2] |> Flux.Tracker.data, events, 2:3), 3)
    println(" Done.")
  
    print("Computing distances...")
    dists = mapslices(norm, preds - points, 2)
    println(" Done.")

    print("Computing statistics...") 
    me = mean(dists)
    mo = mode(dists)
    st = std(dists, mean=me)
    m = minimum(dists)
    M = maximum(dists)
    println(" Done.")

    print("Generating histogram...")
    linhist = stephist(dists, legend=false, title=model_name)
    xmin, xmax = xlims()
    ymin, ymax = ylims()
    xrel(f) = xmin + f*(xmax - xmin)
    yrel(f) = ymin + f*(ymax - ymin)
    annotate!(xrel(0.7), yrel(0.9),
              "Min -- Max: "*string(round(m, 3))*" -- "*string(round(M, 3)))
    annotate!(xrel(0.7), yrel(0.8), "Mean: "*string(round(me, 3)))
    annotate!(xrel(0.7), yrel(0.7), "Mode: "*string(round(mo, 3)))
    annotate!(xrel(0.7), yrel(0.6), "STD: "*string(round(st, 3)))
    loghist = stephist(dists, legend=false, yaxis=(:log10, (1, Inf)),
                       xlabel="Dist. of pred. from true (mm)")
    println(" Done.")

    plot(layout=(2, 1), linhist, loghist)
end

end # module RegCNN
