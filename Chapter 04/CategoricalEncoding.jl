#- Import Packages
using DataFrames

#- Create a DataFrame
df = DataFrame(
  numer = [1, 2, 3, 4],
  categ = ["XL", "XXL", "S", "L"],
)

# 4×2 DataFrame
#  Row │ numer  categ
#      │ Int64  String
# ─────┼───────────────
#    1 │     1  XL
#    2 │     2  XXL
#    3 │     3  S
#    4 │     4  L

# ==============================================================================
# 1. Ordinal Encoding
encodes = Dict("S"=>1, "M"=>2, "L"=>3, "XL"=>4, "XXL"=>5)

# replace values from encodes inplace in the column categ
df.categ = get.(Ref(encodes), df.categ, missing);
# 4×2 DataFrame
#  Row │ numer  categ
#      │ Int64  Int64
# ─────┼──────────────
#    1 │     1      4
#    2 │     2      5
#    3 │     3      1
#    4 │     4      3

transform!(df, Cols(:categ) => x -> get.(Ref(encodes), x, missing), renamecols=false)
# 4×2 DataFrame
#  Row │ numer  categ
#      │ Int64  Int64
# ─────┼──────────────
#    1 │     1      4
#    2 │     2      5
#    3 │     3      1
#    4 │     4      3

#* Usin BetaML
using BetaML, DataStructures
encodes = OrderedDict("S"=>1, "M"=>2, "L"=>3, "XL"=>4, "XXL"=>5)
model = OrdinalEncoder(handle_unknown="infrequent")
fit!(model, identity.(keys(encodes)))
transform!(df, Cols(:categ) => x -> predict(model, df.categ), renamecols=false)
# 4×2 DataFrame
#  Row │ numer  categ
#      │ Int64  Int64
# ─────┼──────────────
#    1 │     1      4
#    2 │     2      5
#    3 │     3      1
#    4 │     4      3

#* Usin CategoricalArrays
using CategoricalArrays
df.categ = levelcode.(
  CategoricalArray(
    df.categ, ordered=true, levels=["S", "M", "L", "XL", "XXL"]
  )
)
# 4×2 DataFrame
#  Row │ numer  categ
#      │ Int64  Int64
# ─────┼──────────────
#    1 │     1      4
#    2 │     2      5
#    3 │     3      1
#    4 │     4      3

# ==============================================================================
# 2. Label Encoding
df = DataFrame(
  numer = [1, 2, 3, 4],
  categ = ["class1", "class2", "class1", "class2"]
)
# 4×2 DataFrame
#  Row │ numer  categ
#      │ Int64  String
# ─────┼───────────────
#    1 │     1  class1
#    2 │     2  class2
#    3 │     3  class1
#    4 │     4  class2

class_label = Dict(label=>idx for (idx,label)∈enumerate(Set(df.categ)))
transform!(df, Cols(:categ) => x -> get.(Ref(class_label), x, missing), renamecols=false)
# 4×2 DataFrame
#  Row │ numer  categ
#      │ Int64  Int64
# ─────┼──────────────
#    1 │     1      1
#    2 │     2      2
#    3 │     3      1
#    4 │     4      2

df.categ = get.(Ref(class_label), df.categ, missing)
# 4×2 DataFrame
#  Row │ numer  categ
#      │ Int64  Int64
# ─────┼──────────────
#    1 │     1      1
#    2 │     2      2
#    3 │     3      1
#    4 │     4      2

#* Usin CategoricalArrays
using CategoricalArrays
df.categ = levelcode.(categorical(df.categ))
# 4×2 DataFrame
#  Row │ numer  categ
#      │ Int64  Int64
# ─────┼──────────────
#    1 │     1      1
#    2 │     2      2
#    3 │     3      1
#    4 │     4      2
# ==============================================================================
# 3. One-Hot Encoding
df = DataFrame(
  color = ["red", "green", "blue"],
  x = [1, 2, 3]
)
# 3×2 DataFrame
#  Row │ color   x
#      │ String  Int64
# ─────┼───────────────
#    1 │ red         1
#    2 │ green       2
#    3 │ blue        3

transform!(df, [:color => ByRow(==(c)) => c for c in unique(df.color)])
# 3×5 DataFrame
#  Row │ color   x      red    green  blue
#      │ String  Int64  Bool   Bool   Bool
# ─────┼────────────────────────────────────
#    1 │ red         1   true  false  false
#    2 │ green       2  false   true  false
#    3 │ blue        3  false  false   true

function OneHotEncod(vec::Vector{String})
  vec .== permutedims(unique(vec))

  #* If I nead Float64 values
  # convert(Matrix{Float64}, vec .== permutedims(unique(vec)))

  # reduce(hcat, [vec .== i for i=unique(vec)])
end

transform!(df, Cols(:color) => OneHotEncod => AsTable)
# 3×5 DataFrame
#  Row │ color   x      x1     x2     x3
#      │ String  Int64  Bool   Bool   Bool
# ─────┼────────────────────────────────────
#    1 │ red         1   true  false  false
#    2 │ green       2  false   true  false
#    3 │ blue        3  false  false   true

#* Using ScikitLearn
using ScikitLearn
using DataFrames
@sk_import preprocessing: OneHotEncoder
@sk_import compose: ColumnTransformer

encoder = OneHotEncoder(categories="auto", drop="first")
transformer = ColumnTransformer([
  ("onehot", encoder, [0]),
  ("nothing", "passthrough", [1])
])

encoded = fit_transform!(transformer, Matrix(df))
encoded = convert(Matrix{Float64}, encoded)

df = DataFrame(
  encoded,
  ["red", "green", "x"]
)
# 3×3 DataFrame
#  Row │ red      green    x
#      │ Float64  Float64  Float64
# ─────┼───────────────────────────
#    1 │     0.0      1.0      1.0
#    2 │     1.0      0.0      2.0
#    3 │     0.0      0.0      3.0

#* Using PyCall
using PyCall

py"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def oneHotEncode(data, idx_col:int):
  # a zero based index function
  # If you want to set the first column as the one that sould be transformed
  # pass 0 as the second positional argument

  idx_other_cols = [idx for idx in range(1, data.shape[1]) if idx != idx_col]

  encoder = OneHotEncoder(categories="auto", drop="first")
  transformer = ColumnTransformer([
    ("onehot", encoder, [idx_col]),
    ("nothing", "passthrough", idx_other_cols)
  ])

  encoded = transformer.fit_transform(data)
  return encoded
"""

encoded = py"oneHotEncode"(Matrix(df), 0)
df = DataFrame(
  encoded,
  ["red", "green", "x"]
)
# 3×3 DataFrame
#  Row │ red      green    x
#      │ Float64  Float64  Float64
# ─────┼───────────────────────────
#    1 │     0.0      1.0      1.0
#    2 │     1.0      0.0      2.0
#    3 │     0.0      0.0      3.0

# ==============================================================================
# 4. Frequency Encoding
df = DataFrame(
  categ = ["a", "a", "a", "b"],
  numer = [1, 2, 3, 4],
)
# 4×2 DataFrame
#  Row │ categ   numer
#      │ String  Int64
# ─────┼───────────────
#    1 │ a           1
#    2 │ a           2
#    3 │ a           3
#    4 │ b           4
transform(
  df,
  names(df, String) .=> (
    x -> select(
      groupby(DataFrame(x=x), :x),
      proprow,
      keepkeys=false
    ).proprow
  ),
  renamecols=false)
#   4×2 DataFrame
#   Row │ categ    numer
#       │ Float64  Int64
#  ─────┼────────────────
#     1 │    0.75      1
#     2 │    0.75      2
#     3 │    0.75      3
#     4 │    0.25      4
