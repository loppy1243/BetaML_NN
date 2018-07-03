export RegCNN
module RegCNN

using Flux
using BetaML_Data
using ..BetaML_NN

regloss() = model -> (event, point) -> norm(model(event) - point)

convdense(activ; η=0.1) =
    Model(Chain(x -> pad(x/MAX_E),
                x -> reshape(x, (GRIDSIZE+[2, 2])..., 1, 1),
                Conv((3, 3), 1=>1, first(activ)),
                x -> reshape(x, CELLS),
                Dense(CELLS, 2)),
          regloss,
          x -> SGD(x, η),
          :activ => last(activ), :η => η, :opt => "SGD")

fullconvloss() = nothing
function fullconv(activ)
    conv_layer = Chain(fill(Conv((3, 3), 1=>1, first(activ)), 6)...,
                       Conv((2, 2), 1=>1),
                       x -> reshape(x, 9))
    model = x -> (Chain(conv_layer, softmax, x -> reshape(x, 3, 3))(event),
                  Chain(conv_layer, x -> first(activ).(x), Dense(9, 2))(event))
    
    Model(model, fullconvloss, x -> SGD(x, η),
          :activ => last(activ), :η => η, :opt => "SGD")
end

train(modelfile, model) = train(modelfile)

end # module RegCNN
