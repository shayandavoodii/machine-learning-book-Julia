# ===============================================================================
#- Import packages
using DataFrames, Statistics, Impute

# ===============================================================================
#- Create a DataFrame
df = DataFrame(
  a=[1,2,3,4,missing],
  b=[1, missing, 3, 4, missing],
  c=[1, 2, missing, missing, missing],
)
# 5×3 DataFrame
#  Row │ a        b        c
#      │ Int64?   Int64?   Int64?
# ─────┼───────────────────────────
#    1 │       1        1        1
#    2 │       2  missing        2
#    3 │       3        3  missing
#    4 │       4        4  missing
#    5 │ missing  missing  missing

# ===============================================================================
#- Mean Impute each column
DataFrame(
  a=coalesce.(df.a, mean(skipmissing(df.a))),
  b=coalesce.(df.b, mean(skipmissing(df.b))),
  c=coalesce.(df.c, mean(skipmissing(df.c))),
)
# 5×3 DataFrame
#  Row │ a     b        c
#      │ Real  Real     Real
# ─────┼─────────────────────
#    1 │  1    1         1
#    2 │  2    2.66667   2
#    3 │  3    3         1.5
#    4 │  4    4         1.5
#    5 │  2.5  2.66667   1.5

#? Additional exp
# julia> coalesce.([1, missing, 5], 5)
# 3-element Vector{Int64}:
#  1
#  5
#  5

mapcols(x -> coalesce.(x, mean(skipmissing(x))), df)
# 5×3 DataFrame
#  Row │ a     b        c
#      │ Real  Real     Real
# ─────┼─────────────────────
#    1 │  1    1         1
#    2 │  2    2.66667   2
#    3 │  3    3         1.5
#    4 │  4    4         1.5
#    5 │  2.5  2.66667   1.5

transform(df, [:a, :b, :c] .=> x -> coalesce.(x, mean(skipmissing(x))))
# 5×6 DataFrame
#  Row │ a        b        c        a_function  b_function  c_function
#      │ Int64?   Int64?   Int64?   Real        Real        Real
# ─────┼───────────────────────────────────────────────────────────────
#    1 │       1        1        1         1       1               1
#    2 │       2  missing        2         2       2.66667         2
#    3 │       3        3  missing         3       3               1.5
#    4 │       4        4  missing         4       4               1.5
#    5 │ missing  missing  missing         2.5     2.66667         1.5

select(df, [:a, :b, :c] .=> x -> coalesce.(x, mean(skipmissing(x))), renamecols=false)
# 5×3 DataFrame
#  Row │ a     b        c
#      │ Real  Real     Real
# ─────┼─────────────────────
#    1 │  1    1         1
#    2 │  2    2.66667   2
#    3 │  3    3         1.5
#    4 │  4    4         1.5
#    5 │  2.5  2.66667   1.5

# ===============================================================================
#- Median Impute each column
#* Would be the same approaches as above, but using median(skipmissing(x)) instead

# ===============================================================================
#- KNN Impute each column
DataFrame(
  Impute.knn(convert(Matrix{Union{Missing, Float64}}, Matrix(df)), dims=:cols),
  Symbol.(names(df))
)
# 5×3 DataFrame
#  Row │ a         b         c
#      │ Float64?  Float64?  Float64?
# ─────┼──────────────────────────────
#    1 │      1.0       1.0       1.0
#    2 │      2.0       3.0       2.0
#    3 │      3.0       3.0       2.0
#    4 │      4.0       4.0       5.0
#    5 │      4.0       4.0       8.0

#* Bonus from Impute.jl
Impute.filter(df; dims=:rows)
# 1×3 DataFrame
#  Row │ a      b      c
#      │ Int64  Int64  Int64
# ─────┼─────────────────────
#    1 │     1      1      1
