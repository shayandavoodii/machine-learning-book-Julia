using Distributions, MLDatasets, DataFrames, Plots

struct Perceptron
    eta::Float64
    n_iter::Int64
end

"""
The fit function fits a perceptron learning algorithm to the training data.

Parameters
----------
X : {Matrix}, size = [n_samples, n_features]
    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

y : Vector, size = [n_samples]
    Target values.


Returns
-------

fitted object

"""
function fit(perceptron::Perceptron, X::Matrix, y::Vector)

    weights = rand(Normal(0, 0.01), size(X, 2))
    bias = 0.0
    errors = zeros(perceptron.n_iter)

    for epoch∈1:perceptron.n_iter
        error = 0
        for (xᵢ, target)∈zip(collect(eachrow(X)), y)
            update = perceptron.eta * (target - predict(transpose(collect(xᵢ)), weights, bias))
            weights += update * xᵢ
            bias += update
            error += Int(update != 0.0)
        end
        errors[epoch] = error
    end

    return errors
end


function net_input(X, weights, bias::Float64)
    return (X*weights) + bias
end

function predict(X, weights::Vector, bias::Float64)
    return net_input(X, weights, bias)>=0 ? 1 : 0
end

df = Iris(as_df=true).dataframe
X = df[1:100, [1, 3]]
Y = df[1:100, end]
y = convert(Vector{Int},
            replace(x-> x=="Iris-setosa" ? 1 : 0, Array{Union{String, Integer}}(string.(Y)))
)

myp = Perceptron(0.1, 10)
errors = fit(myp, Matrix(X), y)
plot(errors; label="error", marker="o", xlabel="Epochs", ylabel="Number of updates", dpi=300)
