def agrupar(ficheros: list):
    df = pd.DataFrame()
    lista = []
    for archivo in ficheros:
        name, aux = tratamiento(archivo)
        aux = aux.rename({'unidades':name}, axis=1)
        df = pd.concat([df, aux], axis=1)
        lista.append(name)
    return df, lista

df, names = agrupar(['data/filtros.csv', 'data/baterias.csv', 'data/aceites.csv', 'data/limpiaparabrisas.csv'])
