export CatCNN
module CatCNN

import ..BetaML_NN

using Flux
using BetaML_Data

using ..pad, ..catloss, ..Model, ..MAX_E
twolayer(activ; ch=1, ϵ=1, λ=1, η=0.1) =
    Model(Chain(x -> pad(x/MAX_E, width=2),
                x -> reshape(x, (GRIDSIZE+[4, 4])..., 1, 1),
                Conv((3, 3), 1=>ch, first(activ)),
                Conv((3, 3), ch=>1),
                x -> reshape(x, CELLS),
                softmax,
                x -> reshape(x, GRIDSIZE...)),
          catloss(ϵ=1, λ=1),
          x -> SGD(x, η),
          :ch => ch, :activ => last(activ), :ϵ => ϵ, :λ => λ, :η => η)
onelayer(; ϵ=1, λ=1, η=0.1) =
    Model(Chain(x -> pad(x/MAX_E),
                x -> reshape(x, (GRIDSIZE+[2, 2])..., 1, 1),
                Conv((3, 3), 1=>1),
                x -> reshape(x, CELLS),
                softmax,
                x -> reshape(x, GRIDSIZE...)),
          catloss(ϵ=ϵ, λ=ϵ),
          x -> SGD(x, η),
          :ϵ => ϵ, :λ => λ, :η => η)

const model1 = twolayer(relu=>"relu"; ch=1, ϵ=1, λ=1, η=0.1)
const model2 = onelayer(ϵ=0.1, λ=1, η=0.001)

using ..readdata, ..load, ..TRAIN_RANGE
function train(modelfile, model)
    events, points = readdata(TRAIN_RANGE)
    print("Converting points -> cells...")
    cells = mapslices(pointcell, points, 1)
    println(" Done.")

    model, epoch = isfile(modelfile) ? load(modelfile) : (model, 1)

    BetaML_NN.train(modelfile, model, events, cells)
end

end # module CatCNN
