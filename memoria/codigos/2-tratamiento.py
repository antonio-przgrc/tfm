def tratamiento(fichero):
    dataframe = pd.read_csv(fichero)
    dataframe = dataframe.rename({'clave1':'fecha', 'uniTotal':'unidades'}, axis=1)
    dataframe['fecha'] = pd.to_datetime(dataframe['fecha'], format="%Y%m%d")
    dataframe = dataframe.set_index(['fecha'])
    dataframe = dataframe.resample('D').first()
    dataframe.fillna(value=0, inplace=True)
    dataframe = dataframe.groupby(pd.Grouper(freq='B'))
    dataframe = dataframe.sum()
    dataframe[dataframe['unidades'] < 0] = 0
    dataframe = dataframe["2012-01-01":"2024-06-30"]
    name = fichero[fichero.find('/')+1:fichero.find('.')]
    return name, dataframe
