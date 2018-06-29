module CatCNN

# Alternative: create zero array of correct size and fill in with A.
function pad(A, val=0; width=1)
    ret = A
    dims = size(A) |> collect
    off = zeros(Int, ndims(A))
    for i = 1:length(dims)
        ds = dims + off
        zs = fill(val, ds[1:i-1]..., 1, ds[i+1:end]...)
        for _ = 1:width
            ret = cat(i, zs, ret)
            ret = cat(i, ret, zs)
        end
        off[i] += 2width
    end

    ret
end

# Retest
catcnnmodel(activ) = 
    Chain(x -> pad(x),
          x -> reshape(x, (GRIDSIZE+[2, 2])..., 1, 1),
          Conv((3, 3), 1=>1, activ),
          x -> reshape(x, CELLS),
          Dense(CELLS, CELLS),
          x -> reshape(x, GRIDSIZE...),
          softmax)

cat_pure_cnnmodel(ch, activ) =
    Chain(x -> pad(x/MAX_E, width=2),
          x -> reshape(x, (GRIDSIZE+[4, 4])..., 1, 1),
          Conv((3, 3), 1=>ch, activ),
          Conv((3, 3), ch=>1),
          x -> reshape(x, CELLS),
          softmax,
          x -> reshape(x, GRIDSIZE...))

cat_pure_cnnmodel_1(activ) =
    Chain(x -> pad(x/MAX_E),
          x -> reshape(x, (GRIDSIZE+[2, 2])..., 1, 1),
          Conv((3, 3), 1=>1),
          x -> reshape(x, CELLS),
          softmax,
          x -> reshape(x, GRIDSIZE...))

function catcnnmodel_train_main()
    print("Reading data from \"", DATAFILE, "\"...")
    events, inits = readdata(DATAFILE, TRAIN_RANGE) .|> x -> convert(Array{Float64}, x)
    cells = mapslices(pointcell, inits[2:3, :], 1)
    println(" Done.")

    if isfile(MODELFILE)
        print("Loading model from \"", MODELFILE, "\"...")
        BSON.@load MODELFILE model epoch
        println(" Done.")
    else
        model, epoch = cat_pure_cnnmodel_1(relu), 1
    end

    println("Training model...")
    train(MODELFILE, model, catloss(model, ϵ=0.01, λ=1), events, cells, lastepoch=epoch,
          opt=x -> SGD(x, 0.001))
end

end # module CatCNN
