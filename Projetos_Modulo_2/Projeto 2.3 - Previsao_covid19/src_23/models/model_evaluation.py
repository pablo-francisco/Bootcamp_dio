from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

def errors_metrics(y_true,y_pred):
    MAE = mean_absolute_error
    MSE = mean_squared_error

    df_temp = pd.DataFrame()
    df_temp[['MAE','RMSE']]= [[MAE(y_true,y_pred), MSE(y_true,y_pred)**(1/2)]]

    return df_temp


def errors_in_models(cumulative_predictions):

    errors_strategies = pd.DataFrame()

    colunas_prediction = cumulative_predictions.columns[1:]

    for column in colunas_prediction:

        predicted_by_strategy = cumulative_predictions[column]
        MAE_temp = mean_absolute_error(cumulative_predictions.iloc[:,0],predicted_by_strategy)
        RMSE_temp = mean_squared_error(cumulative_predictions.iloc[:,0],predicted_by_strategy) ** (1/2)
        errors_strategies[column] = [MAE_temp,RMSE_temp]
        
    errors_strategies.index = ['MAE','RMSE']

    df_melted = errors_strategies.reset_index().melt(id_vars='index', var_name='Strategy', value_name='Values')

    df_melted.rename(columns={'index': 'Metrics'}, inplace=True)

    df_mae = df_melted[df_melted['Metrics'] == 'MAE'].reset_index(drop=True)
    df_rmse = df_melted[df_melted['Metrics'] == 'RMSE'].reset_index(drop=True)

    errors_strategies = df_mae[['Strategy', 'Values']].rename(columns={'Values': 'MAE'}).copy()
    errors_strategies['RMSE'] = df_rmse['Values']

    return errors_strategies