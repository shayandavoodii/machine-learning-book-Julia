#- Load packages
using Distributions, Random, RDatasets, Plots, VectorizedStatistics

#- Functions
Base.@kwdef mutable struct AdalineSGD
  seed::Int64
  n_features::Int64
  weights = rand(MersenneTwister(seed), Normal(0, 0.01), n_features)
  bias = 0.0
  eta::Float32=0.02
end

function _shuffle!(X::AbstractMatrix{T}, y::AbstractVector{T}, X_dims::Int64) where T<:Real
  shuffled_idx = _shuffle(X, dims=X_dims)
  X .= selectdim(X, X_dims, shuffled_idx)
  y .= selectdim(y, 1, shuffled_idx)
end

function _shuffle(data::AbstractMatrix; dims::Int64=1)
  n = axes(data, dims)
  shuffled_idx = shuffle(n)
  return shuffled_idx
end

function _AdalineSGD(model::AdalineSGD; X::Matrix{T}, y::Vector{T}, n_iter::Int64) where T<:Real
  losses = Vector{Float64}(undef, n_iter)

  for iter∈1:n_iter
    _shuffle!(X, y, 1)
    _losses = Vector{Float64}(undef, size(X, 1))
    for idx∈axes(X, 1)
      xᵢ = X[idx, :]
      _net_input = net_input(model, xᵢ)
      _losses[idx] = _update_weights(model, xᵢ, y[idx], _net_input)
    end
    losses[iter] = mean(_losses)
  end
  return losses
end

function net_input(instance::AdalineSGD, X::AbstractVector)
  X'*instance.weights + instance.bias
end

function activation(X)
  return X
end

function _update_weights(instance::AdalineSGD, xᵢ, target, net_input)
  error = target - activation(net_input)
  instance.weights .+= instance.eta * 2. * xᵢ * error
  instance.bias += instance.eta * 2. * error
  return error^2
end

function predict(instance::AdalineSGD, X)
  activation(net_input(instance, X)) >=0.5 ? 1 : 0
end

#- Load Iris
iris = dataset("datasets", "iris");
X = Matrix(iris[1:100, [1, 2]]);
X_std= (X .- mean(X, dims=1)) ./ std(X, dims=1);
y = convert(Vector{Float64}, String.(iris[:, 5]) .== "setosa")[1:100];
# y = (String.(iris[:, 5]) .== "setosa")[1:100];

ada1 = AdalineSGD(seed=123, n_features=2, eta=0.003)
loss = _AdalineSGD(ada1, X=X_std, y=y, n_iter=20)
# @benchmark _AdalineSGD($ada1, X=$X_std, y=$y, n_iter=20)

#- Plot loss
plot(
  loss,
  label="loss",
  xlabel="Epochs",
  ylabel="Mean Squared Error",
  title="Adaline - Learning rate $(ada1.eta)",
  legend=:topright,
  marker=:o,
  dpi=300
)
