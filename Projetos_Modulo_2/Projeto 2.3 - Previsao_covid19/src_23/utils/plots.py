import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from src_23.models.model_evaluation import errors_metrics


def plot_volume_diario(data,largests=10):

    """
    Input: 
        data: tabular data (transformed data)
        largests: Largest data volume displayed in graph

    Output:
        Graphic plot.

    Description:
        Shows the daily volume of data recived by region.

    """

    df_temp = data.copy()
    df_temp1 = df_temp.groupby(['Country/Region']).size().nlargest(largests).to_frame().reset_index()
    df_temp1.columns = list([df_temp1.columns[0]]) +  ['Quantidade de dados']

    fig , axs = plt.subplots(1,1,figsize=(12,6))
    sns.barplot(df_temp1,x='Country/Region',y='Quantidade de dados', ax=axs)
    axs.tick_params(axis='x',labelrotation=90)
    plt.suptitle('Volume de dados diários por região')
    plt.show()

def plot_picos_regiao(data,coluna,y_label,largests=10):


    """
    Input: 
        data: tabular data (transformed data) 
        coluna: Column selected
        y_label: Label of y axis and title
        largests: Largest data peaks displayed in graph

    Output:
        Graphic plot.

    Description:
        Shows the daily peaks of data by region.
    """


    df_temp = data.copy()
    df_temp1 = df_temp.groupby(['Country/Region'])[coluna].max().nlargest(largests).to_frame().reset_index()
    df_temp1.columns = list([df_temp1.columns[0]]) +  [f'Quantidade de {y_label}']

    fig , axs = plt.subplots(1,1,figsize=(12,6))
    sns.barplot(df_temp1,x='Country/Region',y=f'Quantidade de {y_label}', ax=axs)

    axs.tick_params(axis='x',labelrotation=90)
    axs.ticklabel_format(axis = 'y',style='plain')

    plt.suptitle(f'Quantidade de {y_label} por região')
    plt.show()
    

def plot_dados_regiao(data):

    """
    Input: 
        data: tabular data (transformed and filtred data) 

    Output:
        Graphic plot.

    Description:
        Shows the progression of the disease.
    """

    df_temp = data.copy()
    columns_used = df_temp.columns[1:-1]
    last_column = df_temp.columns[-1]

    colors = ['red','blue','black'] * 3

    fig, axs = plt.subplots(4, 3, figsize=(16, 8), dpi=150)


    for ax, column,color in zip(axs.flatten()[:9], columns_used,colors):
        sns.lineplot(df_temp, x='ObservationDate', y=column, ax=ax,color=color)
        ax.tick_params(axis='x', labelrotation=45)
        ax.ticklabel_format(axis='y', style='plain')
        ax.set_xlabel('')
        ax.grid()

    for ax in axs[3, :]:
        ax.set_visible(False)

    ax_big = plt.subplot2grid((4, 3), (3, 0), colspan=3)
    sns.lineplot(df_temp, x='ObservationDate', y=last_column, ax=ax_big,color='green')
    ax_big.tick_params(axis='x', labelrotation=45)
    ax_big.ticklabel_format(axis='y', style='plain')
    ax_big.set_xlabel('')
    ax_big.grid()     

    plt.suptitle(f'Dados cumulativos x diários x taxas de crescimento',fontsize = 14)
    plt.tight_layout()
    plt.show()

def decompor_series(data,coluna):


    """
    Input: 
        data: tabular data (transformed and filtred data)
        coluna: Column selected

    Output:
        Graphic plot.

    Description:
        Shows the time series decomposed in:
            - Observed values
            - Trend
            - Seasonal
            - Residuals
    """

    result = seasonal_decompose(data.set_index('ObservationDate')[coluna])

    timeseries_analysis = [result.observed,
                            result.trend,
                            result.seasonal,
                            result.resid]

    colors = ['blue','orange','green','red']

    fig, axs = plt.subplots(4, 1,figsize=(12,6))

    for ax,series_analysis,color in zip(axs[:-1],timeseries_analysis[:-1],colors[:-1]):
        sns.lineplot(series_analysis,ax=ax,color=color)

    sns.scatterplot(timeseries_analysis[-1],ax=axs[-1],color=colors[-1])
    axs[-1].axhline(0, linestyle='dashed', c='black')

    for ax in axs:
        ax.set_xlabel('')

    fig.suptitle('Análise de componentes da série temporal')
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------
# Model plots

def plotar_resultados(true_values,pred_values,titulo):

    """
    Input: 
        true_values: Real values of time series 
        pred_values: Predicted values of time series 
        titulo: Title of graph

    Output:
        Graphic plot.

    Description:
        Shows the line plot for visualization
        of predicted and real values. Also displays
        the barplot for error metrics between the two.
    """


    error_values = errors_metrics(true_values,pred_values)

    fig, axs = plt.subplots(1,2,figsize=(16,6))

    sns.lineplot(true_values.to_frame(),x=true_values.index,y=true_values.values,ax=axs[0],color='blue',label='Dados (Real)')
    sns.lineplot(pred_values.to_frame(),x=pred_values.index,y=pred_values.values,ax=axs[0],color='orange',label='Dados (Previsto)')

    axs[0].ticklabel_format(axis='y', style='plain')
    axs[0].legend()

    sns.barplot(error_values,ax=axs[1])

    axs[0].set_title("Real x Previstos")
    axs[1].set_title("Métricas de erros")

    plt.suptitle(titulo)

    plt.tight_layout()
    plt.show()

def plot_strategy_predictions(cumulative_predictions):

    """
    Input: 
        cumulative_predictions: Dataframe with real values
            in first column and other predictions in the
            next columns.

    Output:
        Graphic plot.

    Description:
        Shows the line plot for visualization
        of predicted and real values by strategies
    """

    fig , ax = plt.subplots(1,1,figsize=(12,6))

    sns.lineplot(cumulative_predictions,ax=ax)
    ax.tick_params(axis='x',labelrotation=90)
    ax.set_ylabel('N° de pessoas')
    plt.suptitle('Progressão da doença (dados de teste)')
    plt.show()

def plot_strategy_errors(cumulative_errors):

    """
    Input: 
        cumulative_errors: Dataframe with strategies and the
        following erros metrics: MAE and RMSE for each.

    Output:
        Graphic plot.

    Description:
        Shows the  scatterplot for visualization
        of errors by strategies.
    """


    fig , ax = plt.subplots(1,1,figsize=(12,6))

    sns.scatterplot(cumulative_errors,x='MAE',
                    y='RMSE',hue='Strategy',ax=ax)

    ax.grid()

    plt.suptitle('Comparação de erros de métricas por estratégia')
    plt.tight_layout()
    plt.show()