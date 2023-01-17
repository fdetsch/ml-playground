### Three Approaches to Encoding Time Information as Features for ML Models ----
### available online: https://developer.nvidia.com/blog/three-approaches-to-encoding-time-information-as-features-for-ml-models/

## SETUP AND DATA ====

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_absolute_error
from sklego.preprocessing import RepeatingBasisFunction

# for reproducibility
np.random.seed(42)

# generate the DataFrame with dates
range_of_dates = pd.date_range(start="2017-01-01",
                           	end="2020-12-30")
X = pd.DataFrame(index=range_of_dates)

# create a sequence of day numbers
X["day_nr"] = range(len(X))
X["day_of_year"] = X.index.day_of_year

# generate the components of the target
signal_1 = 3 + 4 * np.sin(X["day_nr"] / 365 * 2 * np.pi)
signal_2 = 3 * np.sin(X["day_nr"] / 365 * 4 * np.pi + 365/2)
noise = np.random.normal(0, 0.85, len(X))

# combine them to get the target series
y = signal_1 + signal_2 + noise

# plot
y.plot(figsize=(16,4), title="Generated time series");
plt.show()

results_df = y.to_frame()
results_df.columns = ["actuals"]


### TIME-RELATED FEATURES ====

TRAIN_END = 3 * 365 # i.e. use first 3 years for training


### dummy variables ----
### (also known as "one-hot encoding")

X_1 = pd.DataFrame(
	data=pd.get_dummies(X.index.month, drop_first=True, prefix="month")
)
X_1.index = X.index
X_1

## fit `lm()` to training data w/ dummy data
model_1 = LinearRegression().fit(X_1.iloc[:TRAIN_END],
                             	y.iloc[:TRAIN_END])

results_df["model_1"] = model_1.predict(X_1)
results_df[["actuals", "model_1"]].plot(figsize=(16,4),
                                    	title="Fit using month dummies")
plt.axvline(date(2020, 1, 1), c="m", linestyle="--");
plt.show()


### cyclical encoding with sine/cosine transformation ----

def sin_transformer(period):
	return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

def cos_transformer(period):
	return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

X_2 = X.copy()
X_2["month"] = X_2.index.month

X_2["month_sin"] = sin_transformer(12).fit_transform(X_2)["month"]
X_2["month_cos"] = cos_transformer(12).fit_transform(X_2)["month"]

X_2["day_sin"] = sin_transformer(365).fit_transform(X_2)["day_of_year"]
X_2["day_cos"] = cos_transformer(365).fit_transform(X_2)["day_of_year"]

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(16,8))
X_2[["month_sin", "month_cos"]].plot(ax=ax[0])
X_2[["day_sin", "day_cos"]].plot(ax=ax[1])
plt.suptitle("Cyclical encoding with sine/cosine transformation");
plt.show()

X_2.plot.scatter(x = 'month_sin', y = 'month_cos')
plt.show()

## fit `lm()` to training data w/ cyclic features
X_2_daily = X_2[["day_sin", "day_cos"]]

model_2 = LinearRegression().fit(X_2_daily.iloc[:TRAIN_END],
                             	y.iloc[:TRAIN_END])

results_df["model_2"] = model_2.predict(X_2_daily)
results_df[["actuals", "model_2"]].plot(figsize=(16,4),
                                    	title="Fit using sine/cosine features")
plt.axvline(date(2020, 1, 1), c="m", linestyle="--");
plt.show()


### radial basis functions ----

rbf = RepeatingBasisFunction(n_periods=12,
                         	column="day_of_year",
                         	input_range=(1,365),
                         	remainder="drop")
rbf.fit(X)
X_3 = pd.DataFrame(index=X.index,
               	data=rbf.transform(X))

X_3.plot(subplots=True, figsize=(14, 8),
     	sharex=True, title="Radial Basis Functions",
     	legend=False);
plt.show()

## fit `lm()` to training data w/ radial basis functions
model_3 = LinearRegression().fit(X_3.iloc[:TRAIN_END],
                             	y.iloc[:TRAIN_END])

results_df["model_3"] = model_3.predict(X_3)
results_df[["actuals", "model_3"]].plot(figsize=(16,4),
                                    	title="Fit using RBF features")
plt.axvline(date(2020, 1, 1), c="m", linestyle="--");
plt.show()


## FINAL COMPARISON ====

results_df.plot(title="Comparison of fits using different time-based features",
            	figsize=(16,4),
            	color = ["c", "k", "b", "r"])
plt.axvline(date(2020, 1, 1), c="m", linestyle="--");
plt.show()

## mean absolute error
score_list = []
for fit_col in ["model_1", "model_2", "model_3"]:
	scores = {
    	"model": fit_col,
    	"train_score": mean_absolute_error(
        	results_df.iloc[:TRAIN_END]["actuals"],
        	results_df.iloc[:TRAIN_END][fit_col]
    	),
    	"test_score": mean_absolute_error(
        	results_df.iloc[TRAIN_END:]["actuals"],
        	results_df.iloc[TRAIN_END:][fit_col]
    	)
	}
	score_list.append(scores)

scores_df = pd.DataFrame(score_list)
scores_df
