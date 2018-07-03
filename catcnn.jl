export CatCNN
module CatCNN

using Flux
using Plots
using BetaML_Data
using ..BetaML_NN

catloss(; ϵ=1, λ=1) = model -> (event, startcell) -> begin
    pred = model(event)

    -log(ϵ + max(0, pred[startcell...])) + abs(λ*(sum(pred) - pred[startcell...]))
end

using ..pad
twolayer(activ; ch=1, ϵ=1, λ=1, η=0.1) =
    Model(Chain(x -> pad(x/MAX_E, width=2),
                x -> reshape(x, (GRIDSIZE+[4, 4])..., 1, 1),
                Conv((3, 3), 1=>ch, first(activ)),
                Conv((3, 3), ch=>1),
                x -> reshape(x, CELLS),
                softmax,
                x -> reshape(x, GRIDSIZE...)),
          catloss(ϵ=ϵ, λ=λ),
          x -> SGD(x, η),
          :ch => ch, :activ => last(activ), :ϵ => ϵ, :λ => λ, :η => η,
          :opt => "SGD")
onelayer(; ϵ=1, λ=1, η=0.1) =
    Model(Chain(x -> pad(x/MAX_E),
                x -> reshape(x, (GRIDSIZE+[2, 2])..., 1, 1),
                Conv((3, 3), 1=>1),
                x -> reshape(x, CELLS),
                softmax,
                x -> reshape(x, GRIDSIZE...)),
          catloss(ϵ=ϵ, λ=ϵ),
          x -> SGD(x, η),
          :ϵ => ϵ, :λ => λ, :η => η, :opt => "SGD")

const model1 = twolayer(relu=>"relu"; ch=1, ϵ=0.1, λ=1, η=0.01)
const model2 = onelayer(ϵ=0.1, λ=1, η=0.001)

train(modelfile, model) = train(modelfile, model, BetaML_NN.readdata(TRAIN_RANGE)...)
function train(modelfile, model, events, points)
    print("Converting points -> cells...")
    cells = mapslices(pointcell, points, 2)
    println(" Done.")

    model, epoch = isfile(modelfile) ? BetaML_NN.load(modelfile) : (model, 1)

    BetaML_NN.train(modelfile, model, events, cells)
end

predict(model, event) = ind2sub(event, applymodel(model, event) |> indmax) |> collect
predict(model) = event -> predict(model, event)

@modelfunc function xyr_hists(model::Model, events, cells; model_name="")
    preds = squeeze(mapslices(predict(model), events, 2:3), 3)

    diff = preds - cells
    xs, ys = diff[:, 1], diff[:, 2]
    rs = mapslices(norm, diff, 2)

    plot(layout=(3, 1),
         histogram(xs, legend=false, xlabel="x diff", title=model_name,
                   yaxis=(:log10, (1, Inf))),
         histogram(ys, legend=false, xlabel="y diff", yaxis=(:log10, (1, Inf))),
         histogram(rs, legend=false, xlabel="radial diff", yaxis=(:log10, (1, Inf))))
end

@modelfunc function accuracies(model::Model, events, cells; model_name="")
    numevents = size(events, 1)

    preds = squeeze(mapslices(predict(model), events, 2:3), 3)

    accs = map(0:8) do i
        count(all(abs.(preds - cells) .<= [i i], 2)) / numevents
    end

    (accs, histogram(accs, legend=false, xlabel="Accuracy", title=model_name))
end

end # module CatCNN
