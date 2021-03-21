import pandas as pd
import numpy as np
from math import log

from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA
from statsmodels.stats.stattools import durbin_watson

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt

METRICS = [lambda real, predictions: np.sqrt(sum((predictions - real.values) ** 2) / len(real)),
           lambda real, predictions: sum((predictions - real.values) ** 2), mean_squared_error, mean_absolute_error,
           lambda real, predictions: np.mean(np.abs((real - predictions) / real)), r2_score
           ]

METRICS_TITLES = ['RMSE', 'RSS', 'MSE', 'MAE', 'MAPE', 'R2']


def aic(n, mse, num_params):
    return n * log(mse) + 2 * num_params


def bic(n, mse, num_params):
    return n * log(mse) + num_params * log(n)


def stationarity_test(ts, stat_test):
    result = stat_test(ts)
    test_name = stat_test.__name__

    test_result = {
        test_name + "_statistics": result[0],
        test_name + "_p_value": result[1],
        test_name + "_critical_values": result[4] if test_name == 'adfuller' else result[
            3] if test_name == 'kpss' else None
    }

    return test_result


def plot_decomposition(ts, model, period, figsize=(30, 20)):
    plt.rcParams.update({'figure.figsize': figsize})

    sesonal_dec = seasonal_decompose(ts, model=model, period=period)
    sesonal_dec.plot()
    plt.show()

    return sesonal_dec


def to_stationary(df, target_column='Close'):
    ts_log = np.log(df[target_column])
    ts_diff = ts_log.diff(periods=1).dropna()
    stationary_ts = ts_diff.diff(periods=1).dropna()

    return stationary_ts, ts_diff, ts_log


def plot_pacf_acf(input_ts, lags=12, figsize=(12, 8)):
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(input_ts, lags=lags, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(input_ts, lags=lags, ax=ax2)


def plot_results(stationary_ts, fitted_values, figsize=(8, 10)):
    stationary_ts.plot(figsize=figsize);
    fitted_values.plot(figsize=figsize);
    plt.legend(['Real time serie', 'Predicted time serie']);
    plt.title('Real and Predicted values for given time series');


def to_source_ts(ts_diff, ts_log, arma_model, train=True):
    if train:
        preds_diff = pd.Series(arma_model.fittedvalues, copy=True)
    else:
        preds_diff = pd.Series(arma_model.forecast()[0][1:], copy=True)

    preds = ts_diff.add(preds_diff, fill_value=0)
    preds = np.exp(ts_log.add(preds, fill_value=0))

    return preds


def calculate_metrics(predictions, real, metrics=METRICS, metrics_titles=METRICS_TITLES):
    metrics_df = pd.DataFrame()

    for metric, metric_title in zip(metrics, metrics_titles):
        calculated_metric = metric(real, predictions)
        metrics_df.loc[0, metric_title] = calculated_metric

    #         print('{}: {:10.4f}'.format(metric_title, calculated_metric))

    return metrics_df


def plot_data_by_fold(df, fold_ids, figsize=(10, 30)):
    fig, axs = plt.subplots(nrows=len(fold_ids), ncols=1, figsize=figsize)

    for idx, folds in enumerate(fold_ids):
        train_df_tmp = df.loc[folds[0], :]
        validate_df_tmp = df.loc[folds[1], :]

        axs[idx].plot(
            train_df_tmp['Date'], train_df_tmp['Close'], "b",
        )
        axs[idx].plot(
            validate_df_tmp['Date'], validate_df_tmp['Close'], "r",
        )
        axs[idx].set_title('Train dates: {} - {}, Test dates: {} - {}'.format(
            train_df_tmp.Date.min(), train_df_tmp.Date.max(), validate_df_tmp.Date.min(), validate_df_tmp.Date.max()))


def calc_all_metrics(train, test, train_preds, test_preds, model, target_column='Close'):

    metrics_df = calculate_metrics(train_preds, train[target_column])
    metrics_df['durbin_watson'] = durbin_watson(model.resid)
    metrics_df['aic'] = aic(train.shape[0],
                            mean_squared_error(train[target_column], train_preds),
                            model.params.shape[0]
                            )
    metrics_df['bic'] = bic(train.shape[0],
                            mean_squared_error(train[target_column], train_preds),
                            model.params.shape[0]
                            )

    metrics_df_test = calculate_metrics(test_preds, test[target_column])
    metrics_df_test['durbin_watson'] = durbin_watson(model.resid)
    metrics_df_test['aic'] = aic(test.shape[0],
                                 mean_squared_error(test[target_column], test_preds),
                                 model.params.shape[0]
                                 )
    metrics_df_test['bic'] = bic(test.shape[0],
                                 mean_squared_error(test[target_column], test_preds),
                                 model.params.shape[0]
                                 )

    return metrics_df, metrics_df_test


def make_cross_validation(df, target_column='Close', n_splits=5, test_size=30, ar_order=(1, 0), arma_order=(1, 2),
                          lags=12,
                          folds_plots_size=(10, 30), stat_plot_size=(10, 8), plot=False
                          ):
    ts = df[target_column]

    folds_indexes = list(TimeSeriesSplit(n_splits=n_splits, test_size=test_size).split(ts))
    min_fold_len = min([len(fold[0]) for fold in folds_indexes])

    for i in range(len(folds_indexes)):
        folds_indexes[i] = (folds_indexes[i][0][-min_fold_len:], folds_indexes[i][1])

    if plot:
        plot_data_by_fold(df, folds_indexes, folds_plots_size)

    all_metrics_df_arma = pd.DataFrame()
    all_metrics_df_test_arma = pd.DataFrame()

    all_metrics_df_ar = pd.DataFrame()
    all_metrics_df_test_ar = pd.DataFrame()

    for fold_index, fold in enumerate(folds_indexes):
        train_indexes = fold[0]
        test_indexes = fold[1]

        train_df_tmp = df.loc[train_indexes, :]
        test_df_tmp = df.loc[test_indexes, :]

        df_stationary, data_diffed, data_logged = to_stationary(train_df_tmp)
        df_stationary_test, data_diffed_test, data_logged_test = to_stationary(test_df_tmp)

        stationarity_test(df_stationary, smt.adfuller)
        stationarity_test(df_stationary, smt.kpss)

        stationarity_test(df_stationary_test, smt.adfuller)
        stationarity_test(df_stationary_test, smt.kpss)

        if plot:
            plt.figure(figsize=(10, 8))
            df_stationary.plot(title='Train Stationary process', figsize=stat_plot_size);
            plt.show()

            plt.figure(figsize=(10, 8))
            df_stationary_test.plot(title='Test Stationary process', figsize=stat_plot_size);
            plt.show()

            plt.figure(figsize=(10, 8))
            plot_pacf_acf(df_stationary, lags=lags)
            plt.show()

        ### Побудова моделі Авторегресії
        ar_model = ARMA(df_stationary, order=ar_order)
        ar_model_fitted = ar_model.fit()

        ar_predictions = to_source_ts(data_diffed, data_logged, ar_model_fitted)
        test_ar_predictions = to_source_ts(data_diffed_test, data_logged_test, ar_model_fitted, False)

        metrics_df_ar, metrics_df_test_ar = calc_all_metrics(train_df_tmp, test_df_tmp, ar_predictions,
                                                             test_ar_predictions, ar_model_fitted
                                                             )
        metrics_df_ar.index = pd.Series(str((ar_order[0], ar_order[1], fold_index)))
        metrics_df_test_ar.index = pd.Series(str((ar_order[0], ar_order[1], fold_index)))

        all_metrics_df_ar = pd.concat((all_metrics_df_ar, metrics_df_ar), axis=0)
        all_metrics_df_test_ar = pd.concat((all_metrics_df_test_ar, metrics_df_test_ar), axis=0)

        if plot:
            plt.figure(figsize=(10, 8))
            plot_pacf_acf(ar_model_fitted.resid.dropna(), lags=lags)
            plt.show()

        ### Побудова моделі Авторегресії та ковзного середнього

        try:
            arma_model = ARMA(df_stationary, order=arma_order)
            arma_model_fitted = arma_model.fit()

        except ValueError:
            print('IDX: {}, P: {}, Q: {}'.format(fold_index, arma_order[0], arma_order[1]))
            break

        if plot:
            plt.figure(figsize=(10, 8))
            plot_results(df_stationary, arma_model_fitted.fittedvalues)
            plt.title('Train');
            plt.show()

            plt.figure(figsize=(10, 8))
            plot_results(df_stationary_test, pd.Series(arma_model_fitted.forecast(steps=test_size)[0]))
            plt.title('Test');
            plt.show()

        arma_predictions = to_source_ts(data_diffed, data_logged, arma_model_fitted)
        test_arma_predictions = to_source_ts(data_diffed_test, data_logged_test, arma_model_fitted, False)

        if plot:
            plt.figure(figsize=(10, 8))
            plot_results(train_df_tmp[target_column], arma_predictions)
            plt.show()

            plt.figure(figsize=(10, 8))
            plot_results(test_df_tmp[target_column], test_arma_predictions)
            plt.show()

        metrics_df_arma, metrics_df_test_arma = calc_all_metrics(train_df_tmp, test_df_tmp, arma_predictions,
                                                                 test_arma_predictions, arma_model_fitted
                                                                 )
        metrics_df_arma.index = pd.Series(str((arma_order[0], arma_order[1], fold_index)))
        metrics_df_test_arma.index = pd.Series(str((arma_order[0], arma_order[1], fold_index)))

        all_metrics_df_arma = pd.concat((all_metrics_df_arma, metrics_df_arma), axis=0)
        all_metrics_df_test_arma = pd.concat((all_metrics_df_test_arma, metrics_df_test_arma), axis=0)

    # all_metrics_df_ar = all_metrics_df_ar.reset_index(drop=True)
    # all_metrics_df_ar.index.name = 'Cross-val iteration'
    #
    # all_metrics_df_test_ar = all_metrics_df_test_ar.reset_index(drop=True)
    # all_metrics_df_test_ar.index.name = 'Cross-val iteration'
    #
    # all_metrics_df_arma = all_metrics_df_arma.reset_index(drop=True)
    # all_metrics_df_arma.index.name = 'Cross-val iteration'
    #
    # all_metrics_df_test_arma = all_metrics_df_test_arma.reset_index(drop=True)
    # all_metrics_df_test_arma.index.name = 'Cross-val iteration'

    return all_metrics_df_arma, all_metrics_df_test_arma, all_metrics_df_ar, all_metrics_df_test_ar
