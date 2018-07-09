using StatsBase: mode

macro modelfunc(funcdecl)
    @assert funcdecl.head == :function || funcdecl.head == :(=) #=
         =# && funcdecl.args[1].head == :call

    fname = esc(funcdecl.args[1].args[1])

    quote
        $fname(modelfile::AbstractString, events, cells; kws...) =
            $fname(BetaML_NN.load(modelfile)[:model], events, cells; kws...)

        $(esc(funcdecl))
    end
end

@modelfunc function pointhist(model, events, points; model_name="", batchsize=1000)
    numevents = size(events, 3)
    print("Computing predictions (out of $numevents)...")
    preds = Array{Float64}(2, numevents)
    for i = 0:batchsize:numevents-1
        print(" ", i)
        pred_dists, pred_rel_point_dists = model(events[:, :, i+1:min(end, i+batchsize)]) .|> data
        for j in indices(pred_dists, 3)
            dist = pred_dists[:, :, j]
            pred_cell = ind2sub(dist, indmax(dist))
            pred_cell_point = cellpoint(pred_cell)
            pred_rel_point = pred_rel_point_dists[pred_cell..., :, j]
            preds[:, i+j] .= pred_cell_point + pred_rel_point
        end
    end
    println(" Done.")
  
    print("Computing distances...")
    dists = mapslices(norm, preds - points, 1)
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
