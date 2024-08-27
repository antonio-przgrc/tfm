dfts = df_res_m.reset_index()
for col in dfts.columns:
    if (col == name) or (col == 'fecha'):
        continue

    dfts[col+'_ts'] = 0

    for i in range(0,len(df)):
        errors = dfts[name][0:i+1] - dfts[col][0:i+1]
        bias = np.sum(errors)
        mad = np.mean(np.abs(errors))

        if mad == 0:
            ts = float('inf')
        else:
            ts = bias/mad

        dfts[col+'_ts'][i] = ts

dfts.to_csv(f'results/ts_{name}.csv')