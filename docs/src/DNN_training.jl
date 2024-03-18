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

function getdata(bs)
  ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

  # Loading Dataset	
  xtrain, ytrain = MLDatasets.MNIST(Tx = Float32, split = :train)[:]
  xtest, ytest = MLDatasets.MNIST(Tx = Float32, split = :test)[:]

  # Reshape Data in order to flatten each image into a linear array
  xtrain = Flux.flatten(xtrain)
  xtest = Flux.flatten(xtest)

  # One-hot-encode the labels
  ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

  # Create DataLoaders (mini-batch iterators)
  train_loader = DataLoader((xtrain, ytrain), batchsize = bs, shuffle = true)
  test_loader = DataLoader((xtest, ytest), batchsize = bs)

  return train_loader, test_loader
end

device = gpu
train_loader, test_loader = getdata(128)

## Construct model
model = build_model() |> device

# now we set the model to FluxNLPModel
nlp = FluxNLPModel(model, train_loader, test_loader; loss_f = loss)

n = nlp.meta.nvar
w = nlp.w

max_time = 60. # run at most 1min
callback = (nlp, 
            solver, 
            stats) -> FluxNLPModels.minibatch_next_train!(nlp)

solver_stats = R2(nlp; callback, max_time)
train_acc = FluxNLPModels.accuracy(nlp; data_loader = train_loader)
test_acc = FluxNLPModels.accuracy(nlp) #on the test data