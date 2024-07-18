import pandas as pd
import numpy as np



def feature_engineering(data,group_data=True):

    """
    Input: 
        data: tabular data (DataFrame pandas)
        group_data: Group data by region

    Output:
        New dataframe with daily features and daily rates.

    Description:
        Create new columns (Ex: New confirmed cases, Confirmed growth rate)
        and a adapted reproduction rate for the disease (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7438206/)

    """

    data_temp = data.copy()

    target_features = ['Confirmed','Recovered','Deaths']
    new_data_features = [f'New_{x}' for x in target_features]
    rates_data_features = [f'{x}_rate' for x in target_features]

    if group_data:
        data_temp[new_data_features] = data_temp.groupby(['Country/Region'])[target_features].diff().fillna(0).clip(lower=0)
        data_temp[rates_data_features] = data_temp.groupby(['Country/Region'])[target_features].pct_change().fillna(0)
        data_temp['Reproduction_rate'] = (data_temp['Confirmed_rate']/(data_temp['Recovered_rate'] + data_temp['Deaths_rate'])).replace(np.inf,np.nan).fillna(0)
    else:
        data_temp[new_data_features] = data_temp[target_features].diff().fillna(0).clip(lower=0)
        data_temp[rates_data_features] = data_temp[target_features].pct_change().fillna(0)
        data_temp['Reproduction_rate'] = (data_temp['Confirmed_rate']/(data_temp['Recovered_rate'] + data_temp['Deaths_rate'])).replace(np.inf,np.nan).fillna(0)
   
    return data_temp.replace(np.inf,1)

def filtrar_regiao(data,regiao=None):

    """
    Input: 
        data: tabular data (DataFrame pandas)
        regiao: Selected region to filter

    Output:
        Dataframe with data of the choosen region.

    Description:
        Select the data by region.

    """

    df_temp = data.copy()
    
    if regiao == None:
        return df_temp
    
    else:
        df_temp2 = df_temp[df_temp['Country/Region']==regiao].drop(['Country/Region'],axis=1)
        df_temp2 = df_temp2[df_temp2['Confirmed'] > 0]
        return df_temp2
    
def transformation_pipeline(data,regiao=None):
    """
        Input: 
            data: tabular data (DataFrame pandas)
            regiao: Selected region to filter

        Output:
            Transformed dataframe

        Description:
            Applies the feature engineering techiques and filter the data.


    """
    data_temp = data.copy()

    if regiao in ['global'.lower(),'global'.capitalize(),'global'.upper()]:
        df_global = data_temp.groupby(['ObservationDate'])[['Confirmed','Recovered','Deaths']].sum(numeric_only=True)
        df_transformed = feature_engineering(df_global,group_data=False).reset_index()

    else:
        df_grouped = data_temp.groupby(['Country/Region', 'ObservationDate'])[['Confirmed','Recovered','Deaths']].sum().reset_index()
        df_new_columns = feature_engineering(df_grouped)
        df_transformed = filtrar_regiao(df_new_columns,regiao)
        
    return df_transformed
