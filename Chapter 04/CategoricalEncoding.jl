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

