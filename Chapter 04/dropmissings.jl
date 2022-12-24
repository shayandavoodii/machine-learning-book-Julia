# =============================================================================
#- Import packages
using DataFrames

# =============================================================================
#- Create a DataFrame
df = DataFrame(
  a=[1,2,3,4,missing],
  b=[1, missing, 3, 4, missing],
  c=[1, 2, 3, missing, missing],
)

# 5×3 DataFrame
#  Row │ a        b        c
#      │ Int64?   Int64?   Int64?
# ─────┼───────────────────────────
#    1 │       1        1        1
#    2 │       2  missing        2
#    3 │       3        3        3
#    4 │       4        4  missing
#    5 │ missing  missing  missing

# =============================================================================
#- Drop rows that contain missing values for all columns
df[map(x->!all(ismissing, x), eachrow(df)), :]
# 4×3 DataFrame
#  Row │ a       b        c
#      │ Int64?  Int64?   Int64?
# ─────┼──────────────────────────
#    1 │      1        1        1
#    2 │      2  missing        2
#    3 │      3        3        3
#    4 │      4        4  missing

# Or
deleteat!(df, findall(x->all(ismissing, x), eachrow(df)))
# 4×3 DataFrame
#  Row │ a       b        c
#      │ Int64?  Int64?   Int64?
# ─────┼──────────────────────────
#    1 │      1        1        1
#    2 │      2  missing        2
#    3 │      3        3        3
#    4 │      4        4  missing

# =============================================================================
#- Drop rows that contain missing values for a specific column
dropmissing(df, :a)
# 4×3 DataFrame
#  Row │ a      b        c
#      │ Int64  Int64?   Int64?
# ─────┼─────────────────────────
#    1 │     1        1        1
#    2 │     2  missing        2
#    3 │     3        3        3
#    4 │     4        4  missing

# =============================================================================
#- Drop rows that contain at least one missing value
dropmissing(df, disallowmissing=true)
# 2×3 DataFrame
#  Row │ a      b      c
#      │ Int64  Int64  Int64
# ─────┼─────────────────────
#    1 │     1      1      1
#    2 │     3      3      3

# =============================================================================
#- Deop rows that contain more than 2 missing values
deleteat!(df, findall(x->count(ismissing, x) > 2, eachrow(df)))
# 4×3 DataFrame
#  Row │ a       b        c
#      │ Int64?  Int64?   Int64?
# ─────┼──────────────────────────
#    1 │      1        1        1
#    2 │      2  missing        2
#    3 │      3        3        3
#    4 │      4        4  missing
