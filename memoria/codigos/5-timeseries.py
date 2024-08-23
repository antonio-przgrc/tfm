df = df.reset_index()

scaler = MinMaxScaler(feature_range=(0, 1))

# Definicion TimeSeries
cols = ['tmed','prec', 'hrMedia', 'gasolina', 'diesel']

series = TimeSeries.from_dataframe(df, time_col='fecha', value_cols=names+cols)

series = series.add_holidays(country_code='ES', prov='AN')

transformer = Scaler(scaler)
series = transformer.fit_transform(series)

train, test = series.split_after(pd.Timestamp(year=2023, month=12, day=31))
