using FluxNLPModels
using Flux, NLPModels
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold
using Flux.Losses: logitcrossentropy
using MLDatasets
using JSOSolvers


const loss = logitcrossentropy

function build_model(; imgsize = (28, 28, 1), nclasses = 10)
  return Chain(Dense(prod(imgsize), 32, relu), Dense(32, nclasses)) 
end

function getdata(; T = Float32) #T for types
  ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

  # Loading Dataset
  xtrain, ytrain = MLDatasets.MNIST(Tx = T, split = :train)[:]
  xtest, ytest = MLDatasets.MNIST(Tx = T, split = :test)[:]

  # Reshape Data in order to flatten each image into a linear array
  xtrain = Flux.flatten(xtrain)
  xtest = Flux.flatten(xtest)

  # One-hot-encode the labels
  ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)
  return xtrain, ytrain, xtest, ytest
end

# train_data.features is a 28×28×60000 Array{Float32, 3} of the images.
# Flux needs a 4D array, with the 3rd dim for channels -- here trivial, grayscale.
# Combine the reshape needed with other pre-processing:

function create_batch(; batchsize = 128)
  # Create DataLoaders (mini-batch iterators)
  xtrain, ytrain, xtest, ytest = getdata()
  xtrain = reshape(xtrain, 28, 28, 1, :)
  xtest = reshape(xtest, 28, 28, 1, :)
  train_loader = DataLoader((xtrain, ytrain), batchsize = batchsize, shuffle = true)
  test_loader = DataLoader((xtest, ytest), batchsize = batchsize)
  return train_loader, test_loader
end

train_loader, test_loader = create_batch()

device = gpu # or gpu
model =
  Chain(
    Conv((5, 5), 1 => 6, relu),
    MaxPool((2, 2)),
    Conv((5, 5), 6 => 16, relu),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(256 => 120, relu),
    Dense(120 => 84, relu),
    Dense(84 => 10),
  ) |> device

nlp = FluxNLPModel(model, train_loader, test_loader; loss_f = loss)

n = nlp.meta.nvar
w = nlp.w

max_time = 60. # run at most 1min
callback = (nlp, 
            solver, 
            stats) -> FluxNLPModels.minibatch_next_train!(nlp)

solver_stats = tadam(nlp; callback = callback, max_time = 5.,verbose=1)
train_acc = FluxNLPModels.accuracy(nlp; data_loader = train_loader)
test_acc = FluxNLPModels.accuracy(nlp) #on the test data