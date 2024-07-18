import pandas as pd


def revert_rates(data_rate,last_val):

    """
    Input: 
        data_rate: Dataframe with selected growth rate
        last_val: Last real value to reproduct
          the cumulative data

    Output:
        Dataframe with cumulative data converted.

    Description:
        Converts the growth rate into cumulative data.
    """

    vals_temp = []
    df_temp = pd.DataFrame()
    init_val =( 1 + data_rate.iloc[0,1]) * last_val #coluna 1 para selecionar dados de previstos
    vals_temp.append(init_val)

    for percentual in data_rate.iloc[1:,1]:
        val = (1+percentual) * vals_temp[-1]
        vals_temp.append(val)

    df_temp.index = data_rate.index

    columns_name = data_rate.columns[0].split('_')[0]

    df_temp[f'{columns_name}_r2c'] = vals_temp

    return df_temp

def convert_2_cumulative(train_data,direct_values,daily_values,rates_values):


    """
    Input: 
        train_data: Dataframe with train data
        direct_values: Cumulative values predicted directly
        daily_values: Daily values predicted
        rates_values: Growth rates predicted


    Output:
        Dataframe with cumulative data converted.

    Description:
        Converts the growth rate and daily data into cumulative data.
    """


    final_real_val = train_data.iloc[-1,0] # último valor real usado como entrada
    # para calcular a progressão com dados diários e com percentuais.

    converted_day2cumulative = (daily_values.cumsum().iloc[:,1] + final_real_val).to_frame() #coluna 1 para selecionar dados de previstos
    converted_day2cumulative.columns = [f'{train_data.columns[0]}_d2c']
    
    converted_rate2cumulative = revert_rates(rates_values,final_real_val)
    cumulative_predictions = pd.concat([direct_values,converted_day2cumulative,converted_rate2cumulative],axis=1)
    return cumulative_predictions