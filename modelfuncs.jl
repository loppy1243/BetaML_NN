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

@modelfunc function disthist(model, events, points; model_name="")
    print("Computing predictions...")
    preds = squeeze(mapslices(x -> model(x) |> Flux.Tracker.data, events, 2:3), 3)
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
