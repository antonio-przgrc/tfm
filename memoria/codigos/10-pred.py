resultados = test[name].pd_dataframe()

print(name)
print('prophet')
pred = mod_prophet.predict(n=130)
predicion = pred[name].pd_dataframe().rename({name:'prophet'}, axis=1)
resultados = pd.concat([resultados, predicion], axis=1)

# Se realizan las predicciones de los modelos sin regresor
for i, model in enumerate(models_uni):
    print(model.model_name)
    pred = model.predict(n=130)
    predicion = pred[name].pd_dataframe().rename({name:model.model_name}, axis=1)
    resultados = pd.concat([resultados, predicion], axis=1)

# # Se realizan las predicciones de los modelos con regresor
for i, model in enumerate(models_multi):
    print(model.model_name)
    pred = model.predict(n=130)
    predicion = pred[name].pd_dataframe().rename({name:model.model_name+'_multi'}, axis=1)
    resultados = pd.concat([resultados, predicion], axis=1)

resultados.to_csv(f'results/resultados_{name}.csv')
