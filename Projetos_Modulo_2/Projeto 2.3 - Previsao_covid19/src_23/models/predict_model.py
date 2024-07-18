from statsmodels.tsa.stattools import adfuller
import pandas as pd
from pmdarima.arima import auto_arima
from src_23.models.model_evaluation import errors_metrics
from src_23.utils.plots import plotar_resultados


def dividir_dados(dados,dias_teste=None,por_teste=.3):
    if dias_teste == None:
        ponto_split = int(dados.shape[0]*(1-por_teste))
        X = dados[:ponto_split]
        y = dados[ponto_split:]
    else:
        X = dados[:-dias_teste]
        y = dados[-dias_teste:]
    return X,y


def adf_test(series):
    result = adfuller(series)

    print('-'*150)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')

    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    if result[1] <= 0.05:
        print("A série temporal é estacionária.")
        print('-'*150)
        return True
    else:
        print("A série temporal não é estacionária.")
        print('-'*150)
        return False
    
    
def pipeline_arima(dados,dias_teste=None):

    X, y = dividir_dados(dados,dias_teste=dias_teste)
    stationary = adf_test(X)
    
    model = auto_arima(X, start_p=1, d=None, start_q=1,
                    max_p=5, max_d=5, max_q=5,stationary = stationary)
    
    X_pred = model.predict_in_sample()
    y_pred = model.predict(len(y))

    train_results = pd.DataFrame([X,X_pred]).T
    test_results = pd.DataFrame([y,y_pred]).T
    test_results.columns = [test_results.columns[0],f'{test_results.columns[0]}_direct']
    
    train_error = errors_metrics(X,X_pred)
    test_error = errors_metrics(y,y_pred)

    plotar_resultados(X,X_pred,'Previsão ARIMA - treino')
    plotar_resultados(y,y_pred,'Previsão ARIMA - teste')

    
    return train_results, test_results, train_error, test_error