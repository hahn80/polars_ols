#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import polars as pl
import polars_ols as pls

df = pl.DataFrame(
    {"y": [1.16, -2.16, -1.57, 0.21, 0.22, 1.6, -2.11, -2.92, -0.86, 0.47],
    "x1": [0.72, -2.43, -0.63, 0.05, -0.07, 0.65, -0.02, -1.64, -0.92, -0.27],
    "x2": [0.24, 0.18, -0.95, 0.23, 0.44, 1.01, -2.08, -1.36, 0.01, 0.75],
    "group": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
    })


lasso_expr = pl.col("y").least_squares.lasso(
    "x1", "x2", alpha=0.0001, add_intercept=True
).over("group")

predictions = df.with_columns(
    lasso_expr.round(2).alias("predictions"),
)

print(predictions)

coefficients = df.with_columns(pl.col("y").least_squares.rls(
    pl.col("x1"), pl.col("x2"), mode="coefficients"
).over("group").alias("coefficients"))

print(coefficients)

statistics = (df.select(
    pl.col("group"),
    pl.col("y").least_squares.ols(
    pl.col("x1", "x2"), mode="statistics", add_intercept=True, confidence_level=0.8
   ).over("group")
)
.unnest("statistics")
.explode(["feature_names", "coefficients", "standard_errors", "t_values", "p_values", "lower_bounds", "upper_bounds"])
)

print(statistics.columns)
print(statistics)
