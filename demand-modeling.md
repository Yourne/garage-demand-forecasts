---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.3
  kernelspec:
    display_name: garage-demand-forecasts
    language: python
    name: python3
---

<!-- # Introduction
My family runs a garage, and as a recent MSc Data Science graduate, I wanted to contribute by developing a tool that forecasts how many cars are likely to come in. 
I believe this could help with shift planning and ease the burden caused by the uncertainty of when the garage will return to its normal workload after soem busy hours. -->
# Summary
I developed a model to forecast the hourly demand of parking lots for a seaside garage with multiple seasonlities. The resulting model achieves
an average out-of-sample mean absolute error of 30 lots for with a horizon of 6 hours, 50 lots with a week. The model can be realiably used to help operation managers decide an apt number of workers to accommodate the demand. It is also the first step for a dynamic pricing strategy.

The dataset is novel: I personally collected and cleaned the dataset from the firm.   

## Business requirements
The forecasting horizons are the next few hours and the next few days. Foreseeing the next few hours and the next few days gives the operation manager the possibility to know in advance at different time frame when to call other workers to work (and when to dismiss them). A dynamic pricing strategy can benefit from the same horizons. 

# Dataset
The original dataset consists of a SQL table. Each entry records the datetime of arrival and the datetime of departure the type of vehicle and the paid amount. 

The snippet computes the number of cars for a given frequency

```python
import pandas as pd
def count_cars(
    df: pd.DataFrame, arrival_col: str, departure_col: str, freq_str: str
) -> pd.Series:
    start_date = df[arrival_col].dt.date.min()
    end_date = df[arrival_col].dt.date.max()
    datetime_index = pd.date_range(start_date, end_date, freq=freq_str)
    car_count_ls = list()
    for datetime in datetime_index:
        car_count = sum(
            (datetime > df[arrival_col]) & (datetime < df[departure_col])
        )
        car_count_ls.append(car_count)
    car_count_sr = pd.Series(car_count_ls, index=datetime_index)
    return car_count_sr
```

The frequency chosen is the hour. A shorter time span would increase the computational cost without providing additional value for practical operations. The resulting dataset can be found in `'./data/cars-per-hour.csv`.

```python
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = (12, 4)

cars = pd.read_csv("./data/cars-per-hour.csv", index_col=0, parse_dates=True)
cars.index = pd.DatetimeIndex(cars.index, freq='infer')
cars.plot(figsize=(12, 4), legend=False)
plt.show()
```

The time series shows the hourly car count from May to almost the end of September, 2022. 
There are: 
* a weekly period
* a daily sub-period
* Saturdays and Sundays have higher averages
* there is an increasing trend that peaks in the middle of august, then it dies
* a spike around June 2, and August 15, both a national holidays

```python
cars.groupby(cars.index.month).mean()
```

```python
from matplotlib.dates import DateFormatter
from enum import StrEnum

class Period(StrEnum):
    day = "day"
    week = "week"
    month = "month"
    year = "year"

def plot_subseries(df: pd.DataFrame, column:str, period:Period):
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.freq is not None, "Please set the freq attr of DatetimeIndex"
    assert column in df.columns
    assert period in [Period.week, Period.month]
    weekday_name_ls = [ "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    if period == Period.week:
        n_seasons = 7
        _, ax = plt.subplots(nrows=1, ncols=n_seasons, figsize=(12, 4), sharey=True)
        for i, (_, gb) in enumerate(df.groupby(df.index.day_of_week)):
            if not isinstance(gb.index.freq, pd.tseries.offsets.Day):
                y = gb.groupby(gb.index.normalize()).mean()
            else:
                y = gb
            ax[i].xaxis.set_major_formatter(DateFormatter('%m'))
            ax[i].tick_params("x")
            ax[i].set_title(weekday_name_ls[i])
            ax[i].plot(y)
            sub_series_avg = gb.mean()
            ax[i].hlines(sub_series_avg, y.index[0], y.index[-1], color="blue")
            ax[i].set_ylabel(column)
            ax[i].label_outer()
            ax[i].grid()
    if period == Period.month:
        raise NotImplementedError

plot_subseries(cars, "cars", "week")
```

The red line shows for average number of cars each month and day of the week. The blue line shows the average number of cars for each day of the week.
The plot confirms that:
* non-working days show an average which is higher and double the average of working days.
* June Thursdays have a higher because of June 2 national holiday happened on a Thursday. 
* increasing trend from May to August, especially for working days. 


# Methods


## Modeling with ARMA
Following the Box Jenkins method (Box, G.E.P. and G.M. Jenkins (1970) Time series analysis: Forecasting and control, San Francisco: Holden-Day.), the first question to ask ourselves when modeling with ARMA processes is whether our process is (weakly) stationary.

```python
cars.plot(figsize=(12, 4), legend=False)
plt.show()
```

in our case, it is blatantly non-stationary: 
1. the variance increases up to the middle of August, fading afterward
2. it is weekly periodic, and there is a daily sub-period,

First, we stabilize the variance using the Box-Cox transformation and add one to avoid taking the log of a null value.

```python
from scipy import stats

# compute the optimal lmbda parameter
_, lmbda = stats.boxcox(cars.cars + 1)

cars['boxcox'] = cars.cars \
    .transform(lambda x: x + 1) \
    .transform(lambda x: stats.boxcox(x, lmbda)) 

cars['boxcox'].plot()
plt.show()
print(f'Box-Cox parameter:{lmbda:.4f}')

```

The parameter leans toward 0 rather than 0.5, implying that the variance increases almost quadratically with the trend.

Next, we deseasonalize with period equal to a week.

```python
season = 24 * 7
cars['x'] = cars['boxcox'] - cars['boxcox'].shift(periods=season)
cars['x'].plot()
plt.show()
```

The time series resembles white noise. Let's test whether it needs further differencing.

The augmented Dickey-Fuller statistical test assumes a unit root exits in the time series, implying that the time series must be differenced.

```python
import statsmodels.tsa.api as tsa

p_value = tsa.adfuller(cars.x.dropna())[1]
print(f"ADF test p-value: {p_value:.4f}")
```

The null hypothesis is rejected, so no further differentiation is required.

```python
tsa.graphics.plot_acf(cars.x.dropna())
tsa.graphics.plot_pacf(cars.x.dropna())
plt.show()
```

The sample autocorrelation function (ACF) and sample partial autocorrelation function (PACF) suggests an autoregressive model $AR(p)$, with $p \leq 2$ because:
* the ACF decays expotentially up until the 24-hour lag.
* the PACF are below the 95% threshold after the third lag.

Let's investigate the periodic patterns by extending the time frame to incorporate the weekly period.

```python
tsa.graphics.plot_acf(cars.x.dropna(), lags=24 * 8)
tsa.graphics.plot_pacf(cars.x.dropna(), lags=24 * 8)
plt.show()
```

The extended ACF and PACF show that not all periodic information has been extracted from the time series as the PACF shows elements above threshold values arount 24-hour lag and 168-hour lag.
Let us apply a second order seasonal difference

```python
y = cars.x - cars.x.shift(24*7)
tsa.graphics.plot_acf(y.dropna(), lags=24 * 8)
tsa.graphics.plot_pacf(y.dropna(), lags=24 * 8)
plt.show()
```

The PACF above threshold values are heightened. So the differencing process is subtler than a second order differentiation. Nontheless, let's look at the AR(2) model in-sample evalutation

```python
AR2 = tsa.SARIMAX(cars.x, order=(2, 0, 0), trend='n').fit(disp=False)
AR2.summary()

```

```python
AR2.plot_diagnostics()
plt.tight_layout()
plt.show()
```

On one hand, the ACF of the residuals looks like white noise, hence the $AR(2)$ has good fit. On the other hand, the estimated density function of the residuals show high kurtosis. By looking at the time-series plot we determine that the majority of the samples in the tails belong to the months of May and June and to September, months with a higher variance.

```python
AR2.summary()
```

The Ljung-Box test p-value (prob(Q)) points out that the residuals are white noise, confirming that the model has a good fit.  
On the contrary, the already cited curtosis and the heteroskedasticity's test p-value show that the variance of the process is time-dependent despite of the Box-Cox transformation.

A natural continuation is to model the process using _Generalized Autoregressive Conditional Heteroskedasticity_ (GARCH) models that are designed to handle clusters of variance, but this is a story for another time.

Another attempt to handle high variance is to keep into account outliers using dummy variables. In this case, we can add a dummy variable whether a date is a national holiday and a deterministic time variable. 

```python
holidays_ls = ["2022-05-01", "2022-05-02", "2022-08-15"]

def date_range(date_str: str, n: int, unit: str) -> pd.Series:
    date = pd.to_datetime(date_str)
    range = pd.date_range(date, date +  pd.Timedelta(n, unit=unit), freq=unit)
    return range.to_series()

# date_range('2022-05-01', 23, 'h')
holiday_date_range_sr = pd.concat(
    list(map(lambda x: date_range(x, 23, 'h'), holidays_ls))
    )
cars['is_holiday'] = cars.index.isin(holiday_date_range_sr).astype('int')
```

```python
AR2_ct = tsa.SARIMAX(
    endog=cars.x, 
    # exog=cars.is_holiday, 
    order=(2, 0, 0),
    trend='ct'
    ).fit(disp=False)
AR2_holiday = tsa.SARIMAX(
    endog=cars.x, 
    exog=cars.is_holiday, 
    order=(2, 0, 0),
    trend='ct'
    ).fit(disp=False)
```

```python
print(f"Dynamic regression model with AR(2) errors")
print(f"AIC: {AR2_ct.aic:.4f}, BIC: {AR2_ct.bic:.4f}")
print(f"Dynamic regression model with AR(2) errors")
print(f"AIC: {AR2_holiday.aic:.4f}, BIC: {AR2_holiday.bic:.4f}")
```

As we can see, the increased complexity of the models improves both AIC and BIC.
Among the two, the "ct" (constant and trand) is the best one. 

```python
AR2_ct.summary()
```

```python
AR2_ct.plot_diagnostics()
plt.tight_layout()
plt.show()
```

## Point forecast and prediction interval
just a see its capability

```python
from scipy.special import inv_boxcox

h = 3 * 24

preds_arr = AR2_ct.get_forecast(h).predicted_mean \
    .add(cars.boxcox.tail(season).head(h).values) \
    .transform(lambda x: inv_boxcox(x, lmbda)) \
    .transform(lambda x: x - 1) \
    .transform(round)

conf_int_df = AR2_ct.get_forecast(h).conf_int() \
    .add(cars.boxcox.tail(season).head(h).values.reshape(-1, 1)) \
    .transform(lambda x: inv_boxcox(x, lmbda)) \
    .transform(lambda x: x - 1) \
    .transform(round)


def plot_forecast(y: pd.Series, conf_int: pd.DataFrame | None = None, gt: pd.Series | None = None):
    # plt.figure(figsize=(12, 6))
    plt.plot(y.index, y, label="Predicted Mean")
    if conf_int is not None:
        plt.fill_between(y.index, conf_int["lower x"], conf_int["upper x"], alpha=0.2, label="95 Confidence Interval")
    if gt is not None:
         plt.plot(gt, gt, label="Ground truth", color="orange")
    plt.legend()
    plt.show()

plot_forecast(preds_arr, conf_int_df)
```

## ARMA out-of-sample model selection and evaluation
from the in-sample metrics the best model is a SARIMA(2, 0, 0) with drift.
Now we are going to compute the out-of-sample mean absolute error (MAE) and root mean squared error (RMSE) to evaluate the model performance in a cross-validation fashion adapted to the timeseries context:
the testing algorithm:
1. randomly select N instants of the time series
2. train each model from instant 0 to the selected instant. 
3. forecast h samples 
4. compute root mean squared error and mean absolute error

```python
from collections.abc import Callable
from pprint import pprint
import numpy as np

class ConvergenceError(Exception):
    """Optmization algo did not converge"""
    def __init__(self, message="Failed to converge"):
        self.message = message
        super().__init__(self.message)


def evaluate_forecast_model(
    forecast_model: Callable[[pd.Series, int, float], pd.DataFrame],
    timeseries: pd.Series,
    horizon: int = 24,
    alpha: float = 0.2,
    seed: int = 100,
    split_n: int = 2,
    start_idx: int = 24 * 7 * 2,
):
    """
    Compute the out-of-sample man absolute error (MAE) and root mean squared error 
    (RMSE) over increasing training set sizes (time-series cross-validation).
    The training set are randomly choosen from the input timeseries. 
    """
    assert alpha > 0 and alpha < 0.5
    result_ls = list()
    rng = np.random.default_rng(seed=seed)
    end_idx = len(timeseries) - horizon
    train_idx = rng.choice(np.arange(start_idx, end_idx), split_n, replace=False)
    for idx in np.sort(train_idx):
        train_sr = timeseries.iloc[:idx]
        test_sr = timeseries.iloc[idx : idx + horizon]
        _, lmbda = stats.boxcox(train_sr + 1)
        z = train_sr \
            .transform(lambda x: x + 1) \
            .transform(lambda x: stats.boxcox(x, lmbda))
        try:
            pred_df = forecast_model(z, horizon, alpha) \
                .transform(lambda x: inv_boxcox(x, lmbda)) \
                .transform(lambda x: x - 1) \
                .transform(round)
        except ConvergenceError:
            break

        mae = (test_sr - pred_df.iloc[:, 0]).abs().mean()
        rmse = (test_sr - pred_df.iloc[:, 0]) \
            .transform(lambda x: x**2).mean() ** (1 / 2)
        result_ls.append(
            {
                "model": forecast_model.__name__[:-9], # strip _forecast from the name
                "train_sr_len": idx,
                "alpha": alpha,
                "horizon": horizon,
                "mae": mae,
                "rmse": rmse,
                "predicted_mean": pred_df.iloc[:, 0],
                "lower": pred_df.iloc[:, 1],
                "upper": pred_df.iloc[:, 2],
            }
        )

    result_df = pd.DataFrame(result_ls)
    return result_df

```

```python
# compute point and prediction interval forecasts
# AR(2) + deterministic trend
def sar2_forecast(z: pd.Series, h: int, alpha:float):
    season = 24 * 7
    x = z - z.shift(periods=season)
    model = tsa.SARIMAX(x, order=(2, 0, 0), trend='ct').fit(disp=False)
    if model.mle_retvals['converged'] == False:
        print(model.summary())
        pprint(model.mle_retvals)
        raise ConvergenceError
    mean = model.get_forecast(h).predicted_mean \
        .add(z.tail(season).head(h).values)
    pred_int_df = model.get_forecast(h).conf_int(alpha) \
        .add(z.tail(season).head(h).values.reshape(-1, 1))
    preds = pd.concat([mean, pred_int_df], axis=1)
    return preds

```

```python
HORIZON = 24 * 7
SPLIT_N = 20
ALPHA = 0.2
sar2_week_result_df = evaluate_forecast_model(sar2_forecast, cars.cars, horizon=HORIZON, split_n=SPLIT_N, alpha=ALPHA)
```

The model didn't converge, which is concerning.

The defualt minimization algorithm is the Broyden-Fletcher-Goldfarb-Shanno (BFGS). From the [docs](https://github.com/statsmodels/statsmodels/blob/main/statsmodels/base/model.py), `warnflag: 2` means that _gradient and/or function calls are not changing_ which tipically means that the BFGS algorithm couldn't find a direction of improvement in the objective function, that is, the objective function is flat surface at the given point.

Looking at the model parameters, the drift is almost zero, and the P-value is over ten percent. Let's drop it and see the results

```python
# compute point and prediction interval forecasts
# AR(2) + intercept
def sar2_forecast(z: pd.Series, h: int, alpha:float):
    season = 24 * 7
    x = z - z.shift(periods=season)
    model = tsa.SARIMAX(x, order=(2, 0, 0), trend='c').fit(disp=False)
    if model.mle_retvals['converged'] == False:
        print(f"model did't converge")
        print(model.summary())
        pprint(model.mle_retvals)
        raise ConvergenceError
    mean = model.get_forecast(h).predicted_mean \
        .add(z.tail(season).head(h).values)
    pred_int_df = model.get_forecast(h).conf_int(alpha) \
        .add(z.tail(season).head(h).values.reshape(-1, 1))
    preds = pd.concat([mean, pred_int_df], axis=1)
    return preds

sar2_week_result_df = evaluate_forecast_model(sar2_forecast, cars.cars, horizon=24 * 7, split_n=SPLIT_N, alpha=ALPHA)
sar2_3d_result_df = evaluate_forecast_model(sar2_forecast, cars.cars, horizon=24 * 3, split_n=SPLIT_N, alpha=ALPHA)
sar2_6h_result_df = evaluate_forecast_model(sar2_forecast, cars.cars, horizon=6, split_n=SPLIT_N, alpha=ALPHA)
```

```python
title="Seasonal AR(2) for different horizon and increasing training set size"
ax = pd.DataFrame({
    "horizon: week": sar2_week_result_df["rmse"],
    "horizon: 3d": sar2_3d_result_df["rmse"], 
    "horizon: 6h":  sar2_6h_result_df["rmse"],
    "training set size [weeks]": sar2_6h_result_df["train_sr_len"]
    }).plot(x="training set size [weeks]", ylim=(0, 300), ylabel="mean absolute error", title=title)
ax.set_xticks([24 * 7 * w for w in range(2, 20 + 1, 2)])
ax.set_xticklabels([str(i) for i in range(2, 20 + 1, 2)])
plt.show()
```

Let's wrap up the modeling so far.
The initial transformation are
$$
\begin{split}
z_t &  = boxcox(cars_t + 1 | \lambda) \\
x_t & = (1 - L^m) z_t \quad m = 24 * 7\\
\end{split}
$$
then, we modeled $x_t$ as a second order autoregressive model
$$ (1 - \phi_1 L - \phi_2 L^2)x_t = w_t \quad w \in WN(0, \sigma^2)$$
finally, we added an intercept:
$$ 
y_t = \beta_0 + x_t
$$


# Modeling with Exponential Smoothing
Next, we model the time series within ETS (Error, Trend, Seasonal) theoretical framework. It is challenging because the time-series frequency is hourly, and the seasonal period is weekly, leading to a large number of parameters to optimize and to a potentially unstable optimization routine. 
A typical solution for the large number seasonal parameters is to first remove the seasonal component from the original time-series with a seasonal decomposition method, then fit the ETS model.

The forecast is then obtained as the sum of the deterministic seasonal component and the prediction yielded the ETS.

```python
stl = tsa.STL(cars['boxcox'], period=24 * 7, robust=False).fit()
stl.plot()
plt.show()
```

```python
plt.title("seasonally adjusted ")
cars['s_adj'] = (cars['boxcox'] - stl.seasonal)
cars['s_adj'].plot()
plt.show()
```

```python
ses_a = tsa.ETSModel(cars.s_adj, error='add', trend=None).fit(disp=False)
ses_m = tsa.ETSModel(cars.s_adj, error='mul', trend=None).fit(disp=False)
holt_aa = tsa.ETSModel(cars.s_adj, error='add', trend='add').fit(disp=False)
holt_ma = tsa.ETSModel(cars.s_adj, error='mul', trend='add').fit(disp=False)
holt_am = tsa.ETSModel(cars.s_adj, error='add', trend='mul').fit(disp=False)
holt_mm = tsa.ETSModel(cars.s_adj, error='mul', trend='mul').fit(disp=False)
damped_aa = tsa.ETSModel(cars.s_adj, error='add', trend='add', damped_trend=True).fit(disp=False)
damped_ma = tsa.ETSModel(cars.s_adj, error='mul', trend='add', damped_trend=True).fit(disp=False)
damped_am = tsa.ETSModel(cars.s_adj, error='add', trend='mul', damped_trend=True).fit(disp=False)
damped_mm = tsa.ETSModel(cars.s_adj, error='mul', trend='mul', damped_trend=True).fit(disp=False)

ets_model_ls = [
    ("ses_a", ses_a),
    ("ses_m", ses_m),
    ("holt_aa", holt_aa),
    ("holt_ma", holt_ma),
    ("holt_am", holt_am),
    ("holt_mm", holt_mm),
    ("damped_aa", damped_aa),
    ("damped_ma", damped_ma),
    ("damped_am", damped_am),
    ("damped_mm", damped_mm),
]
```

```python
best_aic_model_idx = pd.Series(m.aic for _, m in ets_model_ls).idxmin()
best_bic_model_idx = pd.Series(m.bic for _, m in ets_model_ls).idxmin()
print(f'best model {ets_model_ls[best_aic_model_idx][0]}, AIC: {ets_model_ls[best_aic_model_idx][1].aic:.2f}')
print(f'best model {ets_model_ls[best_bic_model_idx][0]}, BIC: {ets_model_ls[best_bic_model_idx][1].bic:.2f}')
```

The Akaike's Information Criterion (AIC) and the Bayesian Information Critirion (BIC) agree that the best model is a fitted simple exponential model.
Fun fact: it can be shown that a simple exponential model with additive error is an ARMA(0, 1, 1) [ref](https://people.duke.edu/%7Ernau/411arim.htm). 
<!-- Indeed, simulating such ARMA processes shows a wide variaty of possible paths. Such plethora of future trajectories will be caputured by the prediction intervals. -->

<!-- $$\begin{split}
% (1 - L)y_t & = (1 + \theta_i L)w_t \\
y_t & = y_{t-1} + w_t + \theta_i w_{t-1} \\
\end{split}$$
$$ \begin{split}
s_0 & = x_0 \\
s_t & = \alpha x_t + (1 - \alpha) s_{t-1}  \\ 
\end{split}$$ -->


## Seasonally adjusted ETS model forecast
<!-- $\hat{y}_{T+h|T} = y_{T + h - m(k+1)}$ last seen value in the seasonal period
where $m$ is seasonal period $k = \left \lfloor \frac{h -1}{m} \right \rfloor $ -->
$$\hat{y}_{T+h|T} = y_{T + h - m} + ets(T+h) \quad h \leq m$$
where 
* ets(T+h) is a forecast resulting form that method
* $y_{T + h - m}$ is a naive seasonal forecasting method where the prediction is equal to the last seen season

The package `statsmodels` only offers the ETS forecast, while the naive seasonal forecast must be implemented. The prediction interval are to be computed from package ETSResult class  simulate method as well.
[implementation](https://stackoverflow.com/questions/70277316/how-to-take-confidence-interval-of-statsmodels-tsa-holtwinters-exponentialsmooth)


```python
# point forecasts
# a simple proof of concept
h = 24 * 3

preds = (ses_a.forecast(h) + stl.seasonal.tail(season).head(h).values) \
    .transform(lambda x: inv_boxcox(x, lmbda)) \
    .transform(lambda x: x - 1) \
    .transform(round)

preds.plot()
plt.show()
```

```python
### prediction intervals
from statsmodels.tsa.exponential_smoothing.ets import ETSResults
def compute_pred_interval(ets_model: ETSResults, h: int, alpha:float = 0.05, n_sim: int = 500):
    assert h > 0
    assert alpha > 0 and alpha < 1 
    sim_df = ets_model.simulate(nsimulations=h, repetitions=n_sim, anchor='end')
    upper_level = 1 - alpha
    lower_level = alpha
    upper_ts = sim_df.quantile(upper_level, axis=1)
    lower_ts = sim_df.quantile(lower_level, axis=1)
    pred_int_df = pd.concat([lower_ts, upper_ts], axis=1)
    return pred_int_df

# pred_int_df = compute_pred_interval(ses_a, h)
h = 3 * 24
pred_int_df = compute_pred_interval(ses_a, h, alpha=0.05) \
    .add(stl.seasonal.tail(season).head(h).values.reshape(-1, 1)) \
    .transform(lambda x: inv_boxcox(x, lmbda)) \
    .transform(lambda x: x - 1) \
    .transform(round)
pred_int_df.plot()
plt.show()

```

## Out-of-sample model evaluation
We are going to apply the same cross-valition routine:
1. randomly select N instants of the time series
2. train each model from instant 0 to the selected instant. 
3. forecast h samples 
4. compute root mean squared error and mean absolute error

the choice of the random selection of instants is get a better approximation of the generalization error. 

```python
# compute point and prediction interval forecasts
# seasonal ets
def s_ets_forecast(z: pd.Series, h: int, alpha: float):
    season = 24 * 7
    stl = tsa.STL(z, period=season, robust=False).fit() # seasonal decomposition
    s_adj = z - stl.seasonal # seaonally adjusted
    model = tsa.ETSModel(s_adj, error="add", trend=None).fit(disp=False)
    if model.mle_retvals['converged'] == False:
        print(model.summary())
        pprint(model.mle_retvals)
        raise ConvergenceError
    mean = model.forecast(h)
    sim_df = model.simulate(nsimulations=h, repetitions=500, anchor='end')
    upper_ts = sim_df.quantile(1 - alpha / 2, axis=1)
    lower_ts = sim_df.quantile(alpha / 2, axis=1)
    preds = pd.concat([mean, lower_ts, upper_ts], axis=1) \
        .add(stl.seasonal.tail(season).head(h).values.reshape(-1, 1))
    return preds
```

```python
SPLIT_N = 20
ALPHA = 0.2
s_ets_week_result_df = evaluate_forecast_model(s_ets_forecast, cars.cars, horizon=24 * 7, split_n=SPLIT_N, alpha=ALPHA)
s_ets_3d_result_df = evaluate_forecast_model(s_ets_forecast, cars.cars, horizon=24 * 3, split_n=SPLIT_N, alpha=ALPHA)
s_ets_6h_result_df = evaluate_forecast_model(s_ets_forecast, cars.cars, horizon=6, split_n=SPLIT_N, alpha=ALPHA)
```

# Seasonal AR(2) and Seasonal SES comparison

```python
import matplotlib.dates
def plot_forecast(
    ax: plt.Axes,
    timeseries: pd.Series,
    model: str,
    train_sr_len: int,
    alpha: float,
    horizon: int,
    mae: float,
    rmse: float,
    predicted_mean: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
):
    ax.plot(timeseries.iloc[train_sr_len: train_sr_len + horizon], label="cars")
    ax.plot(predicted_mean, label="predicted mean")
    ax.fill_between(
        predicted_mean.index,
        lower,
        upper,
        label=f"{(1 - alpha)*100:.0f}% prediction interval",
        alpha=0.3,
    )
    ax.set_ylim(bottom = -1, top=600)
    ax.set_title(f"{model} mae {mae:.0f} rmse {rmse:.0f} training samples {train_sr_len}")
    ax.grid()
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d'))
    return ax
```

```python
nrows = len(sar2_week_result_df)
for i in range(nrows):
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12, 4))
    plot_forecast(ax[0], cars.cars, **sar2_week_result_df.iloc[i])
    plot_forecast(ax[1], cars.cars, **s_ets_week_result_df.iloc[i])
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # Make room for the legend
    plt.show()
```


```python
pd.concat([sar2_week_result_df, s_ets_week_result_df, sar2_3d_result_df, s_ets_3d_result_df, sar2_6h_result_df, s_ets_6h_result_df]) \
    .drop(columns=["predicted_mean", "lower", "upper", "train_sr_len", "alpha"]) \
    .groupby(["horizon", "model", ]).mean()
```

Now that we have the full picture, we compare the models.
Both models are not robust to outliers: they translate unpredicted outliers to the following seasonal period  -- see weeks May 15 -> May 22, Aug 16 -> Aug 23, Sept 11 -> Sept 19. 
The SES model outperforms the SAR(2) because the previous week outliers reverberate less over in the prediction. 
Moreover, the SES model predicts better even with two weeks of data. The only advantage of the SAR(2) model is that the quantile forecasts are mostly stable when the horizon distance increases

```python
plt.plot(sar2_6h_result_df["train_sr_len"], sar2_6h_result_df["mae"], label="Seas. AR(2) MAE")
plt.plot(s_ets_6h_result_df["train_sr_len"], s_ets_6h_result_df["mae"], label="Seas. SES MAE")
plt.ylabel("Mean Absolute Error - 6h horizon")
plt.xlabel("training series length in weeks")
plt.ylim(bottom=0, top=200)
plt.xticks([24 * 7 * w for w in range(1, 20)], [str(w) for w in range(1, 20)])
plt.legend()
plt.show()
```

The spikes we see in the MAE coincides with the week with the training size plot correlates with the periods of higher variances. 


# Final considerations and future developments
* The main drawbacks is that not model robust against outliers: an extreme value in the previous week entails a spike in the predicted mean the following week. Yet, from an operational perspective, since the outliers are easily recognizable, the best model can be deployed in production: the avarage error of 30 lots is enough to schedule shifts and create a dynamic pricing strategy.
* From the modeling perspective, GARCH models may overcome the problem of the heteroschedasticity of our time-series. 
* From the analysis perspective, a more systematic approach to evaluate the quantile predictions is computing the _Continuous Ranked Probability Score_ [ref](https://doi.org/10.1146/annurev-statistics-062713-085831)

# Bibliography
* Rob J Hyndman and George Athanasopoulos, _Forecasting: Principles and Practice (3rd ed)_, https://otexts.com/fpp3/
* Robert B. Cleveland,' William S. Cleveland, Jean E. McRae, and Irma Terpenning, _STL: A Seasonal-Trend Decomposition Procedure Based on Loess_, Journal of Official Statistics, Vol. 6. No. 1. 1990. pp. 3-73
* Box, G.E.P. and G.M. Jenkins (1976). _Time Series Analysis, Forecasting and Control. Revised Edition._ Holden Day, San Francisco
* Hamilton, J., (1994), _Applied Time Series Analysis_, Princeton University Press.
* Shumway, R. H. and D. S. Stoffer (2010). _Time Series Analysis and Its Applications: With R Examples_, Springer
