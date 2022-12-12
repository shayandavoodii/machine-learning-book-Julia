#- Load packages
using RDatasets, Random, Statistics, Plots

#- Define AdalineGD
struct AdalineGD
  w::Vector{Float64}
  bias::Float64
  losses::Vector{Float64}
end

function AdalineGD(X, y; eta::Float64=0.01, n_iter::Int64=50, random_state::Int64=1)
  Random.seed!(random_state)
  w = randn(size(X, 2))
  bias = zero(Float64)
  losses = Vector{Float64}(undef, n_iter)

  for i in 1:n_iter
    netInput = net_input(X, w, bias)
    output = activation(netInput)
    errors = y .- output
    w .+= eta * 0.2 * (X' * errors)./size(X, 1)
    bias += eta * 0.2 * mean(errors)
    losses[i] = mean(errors .^ 2)
  end

  return AdalineGD(w, bias, losses)
end

function net_input(X, w, bias)
  return X * w .+ bias
end

function activation(X)
  return X
end

function predict(X, w, bias)
  return activation(net_input(X, w, bias)) .> 0.5 ? 1 : 0
end

#- Load Iris
iris = dataset("datasets", "iris")
X = Matrix(iris[:, 1:4])
y = String.(iris[:, 5])

ada1 = AdalineGD(X, y.=="setosa"; eta=0.01, n_iter=10, random_state=1)

#- Plot loss
plot(
  log10.(ada1.losses),
  label="loss",
  xlabel="Epochs",
  ylabel="Mean Squared Error",
  title="Adaline - Learning rate 0.01",
  legend=:topright
)
