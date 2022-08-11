using Flux, Statistics
using Flux.Data: DataLoader
using Flux: loadmodel!, onehotbatch, onecold, @epochs
using Flux.Losses: mse
using Base: @kwdef
using CUDA
using SymbolicRegression
using StatsBase
using SymbolicUtils
using Symbolics
using PlotlyJS
import Random
using BSON: @save, @load
Random.seed!(42)

@kwdef mutable struct Args
    η::Float64 = 3e-4       ## learning rate
    batchsize::Int = 256    ## batch size
    epochs::Int = 15000        ## number of epochs
    use_cuda::Bool = true   ## use gpu (if cuda available)
end

# create x^2 Data
N = Int(1e4)
iid_intervals = ((-50, 100),)
xs = []
ys = []
n = Int(floor(N / length(iid_intervals)))
for interval in iid_intervals
    x = rand(Float64, n) * (interval[2] - interval[1]) .+ interval[1]
    y = x .^ 2 .+ randn(Float64, n)
    push!(xs, x)
    push!(ys, y)
end
xs = vcat(xs...)
ys = vcat(ys...)

data = hcat(xs, ys)
data_mean = mean(data, dims=1)
data_std = std(data, dims=1)
normalized_data = (data .- data_mean) ./ data_std
xs, ys = normalized_data[:, 1], normalized_data[:, 2]









###### Fitting with overparametrized MLP ######
n_hidden_neurons = 64
function build_model(; nin=(1, 1), nout=1)
    return Chain(Dense(prod(nin), n_hidden_neurons, Flux.relu),
        Dense(n_hidden_neurons, n_hidden_neurons, Flux.relu),
        #BatchNorm(n_hidden_neurons, relu),
        Dropout(0.2),
        Dense(n_hidden_neurons, n_hidden_neurons, Flux.relu),
        #BatchNorm(n_hidden_neurons, relu),
        Dropout(0.2),
        Dense(n_hidden_neurons, nout))
end

function loss(data_loader, model, device)
    ls = 0.0f0
    num = 0
    for (x, y) in data_loader
        x, y = device(x), device(y)
        ŷ = model(x)
        ls += mse(ŷ, y, agg=sum)
        num += size(x)[end]
    end
    return ls / num
end

global train_losses = Float32[]
global test_losses = Float32[]
function train(xs, ys; kws...)
    args = Args(; kws...) ## Collect options in a struct for convenience

    if CUDA.functional() && args.use_cuda
        @info "Training on CUDA GPU"
        CUDA.allowscalar(false)
        device = gpu
    else
        @info "Training on CPU"
        device = cpu
    end


    xtrain = xs[1:Int(N / 2)]
    ytrain = ys[1:Int(N / 2)]
    xtest = xs[Int(N / 2)+1:end]
    ytest = ys[Int(N / 2)+1:end]

    xtrain = Flux.flatten(xtrain)
    xtest = Flux.flatten(xtest)
    ytrain = Flux.flatten(ytrain)
    ytest = Flux.flatten(ytest)

    ## Create test and train dataloaders
    train_loader = DataLoader((xtrain, ytrain), batchsize=args.batchsize, shuffle=true)
    test_loader = DataLoader((xtest, ytest), batchsize=args.batchsize)

    ## Construct model
    model = build_model() |> device
    ps = Flux.params(model) ## model's trainable parameters

    ## Optimizer
    opt = ADAM(args.η)

    ## Training
    for epoch in 1:args.epochs
        for (x, y) in train_loader
            x, y = device(x), device(y) ## transfer data to device
            gs = gradient(() -> mse(model(x), y), ps) #+ sum(sqnorm, Flux.params(model));## compute gradient
            Flux.Optimise.update!(opt, ps, gs) ## update parameters
            global xbatch = x
        end

        ## Report on train and test
        train_loss = loss(train_loader, model, device)
        test_loss = loss(test_loader, model, device)
        push!(test_losses, test_loss)
        push!(train_losses, train_loss)
        println("Epoch=$epoch")
        println("  train_loss = $train_loss")
        println("  test_loss = $test_loss")
    end
    model, test_losses
end


model, test_losses = train(xs, ys)

model_cpu = model |> cpu









######  SymbolicRegression  ######


# xs = reshape(xs, (1, size(xs)...))

# xtrain = xs[:, 1:Int(N / 2)]
# ytrain = ys[1:Int(N / 2)]
# xtest = xs[:, Int(N / 2)+1:end]
# ytest = ys[Int(N / 2)+1:end]

# options = SymbolicRegression.Options(
#     binary_operators=(+, *, /, -),
#     unary_operators=(cos, exp),
#     npopulations=40
# )

# hall_of_fame = EquationSearch(xtrain, ytrain, niterations=100, options=options, numprocs=7)

# dominating = calculate_pareto_frontier(xtrain, ytrain, hall_of_fame, options)

# eqn = node_to_symbolic(dominating[end].tree, options)

# @syms x1

# expression = eqn

# fexp = build_function(expression, x1)

# f = eval(fexp)




#### Plotting #####



newxs = collect(range(-200, 200, 1000))
train_points = PlotlyJS.scatter(x=data[1:Int(N / 2), 1], y=data[1:Int(N / 2), 2], mode="markers", name="Training data, N = $(Int(N/2))")
test_points = PlotlyJS.scatter(x=data[Int(N / 2)+1:end, 1], y=data[Int(N / 2)+1:end, 2], mode="markers", name="Test data, N = $(Int(N/2))")

layout = Layout(title="", xaxis_title="x", yaxis_title="f(x)", legend_title="Legend Title", font=attr(family="Courier New, monospace", size=18, color="RebeccaPurple"))

true_func = PlotlyJS.scatter(x=newxs, y=newxs .^ 2, mode="lines", name="x^2")
normalized_newxs = (newxs .- data_mean[1]) ./ data_std[1]
SRfit = PlotlyJS.scatter(x=newxs, y=f.(normalized_newxs) .* data_std[2] .+ data_mean[2], mode="lines", name="Symbolic regression")
NN_out = model_cpu(normalized_newxs')' .* data_std[2] .+ data_mean[2]
NNfit = PlotlyJS.scatter(x=newxs, y=NN_out[:,1], mode="lines", name="3-layer MLP")

p = PlotlyJS.plot([train_points, test_points, true_func, SRfit, NNfit], layout)

open("./fit.html", "w") do io
    PlotlyBase.to_html(io, p.plot)
end