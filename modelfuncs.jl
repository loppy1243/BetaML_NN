export @modelfunc

using Plots
using StatsBase: mode

macro modelfunc(funcdecl)
    @assert funcdecl.head == :function || funcdecl.head == :(=) #=
         =# && funcdecl.args[1].head == :call

    fname = esc(funcdecl.args[1].args[1])

    quote
        $fname(modelfile::AbstractString, events, cells, args...; kws...) =
            $fname(BetaML_NN.load(modelfile), events, cells, args...; kws...)

        $(esc(funcdecl))
    end
end

@modelfunc function pointhist(model, events, points; model_name="")
    print("Computing predictions...")
    preds = predict(model, events)
    println(" Done.")
  
    print("Computing distances...")
    dists = squeeze(mapslices(norm, preds - points, 1), 1)
    println(" Done.")

    print("Computing statistics...") 
    me = mean(dists)
    mo = mode(dists)
    st = std(dists, mean=me)
    m = minimum(dists)
    M = maximum(dists)
    println(" Done.")

    print("Generating histogram...")
    linhist1 = stephist(dists, legend=false, title=model_name)
    linhist2 = stephist(dists, legend=false, title=model_name, xlims=(0, 4))
    annotate!(xrel(0.7), yrel(0.9),
              "Min -- Max: "*string(round(m, 3))*" -- "*string(round(M, 3)))
    annotate!(xrel(0.7), yrel(0.8), "Mean: "*string(round(me, 3)))
    annotate!(xrel(0.7), yrel(0.7), "Mode: "*string(round(mo, 3)))
    annotate!(xrel(0.7), yrel(0.6), "STD: "*string(round(st, 3)))
    loghist1 = stephist(dists, legend=false, yaxis=(:log10, (1, Inf)),
                        xlabel="Dist. of pred. from true (mm)")
    println(" Done.")

    plot(layout=(1, 2), plot(layout=(2, 1), linhist1, loghist1),
                        plot(layout=(2, 1), linhist2, plot(axis=false)))
end

@modelfunc function quantile(model, events, points, y, p)
    print("Computing predictions...")
    pred_points = predict(model, events)
    println(" Done.")

    print("Computing distances...")
    dists = squeeze(mapslices(norm, points - pred_points, 1), 1)
    println(" Done.")

    print("Computing quantities...")
    x = count(dists .< y)/length(dists)
    q = Base.quantile(dists, p)
    println(" Done.")

    println("(P(<3mm), 90th-%tile) = ", (x, q))

    (x, q)
end
