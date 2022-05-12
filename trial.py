import pickle
def xgb(l:list):  
    df = pd.read_csv("Dataset2.csv")
    loaded_model = pickle.load(open("xbg_model.pkl", 'rb'))
    hour_from =l[0]
    day_from = l[1]
    month_from = l[2]
    year_from = l[3]
    month = df.loc[:,'month'].values
    day = df.loc[:,'day'].values
    year = df.loc[:,'year'].values
    choice= {1:"daily", 2:"weekly", 3:'monthly'}



    df1 = df.loc[(day == day_from)].copy()
    if(df1.shape[0]!=0):
        # print("hello")
        df2 = df1[(df1.loc[:,'month'].values ==month_from)]
    else: 
        df2 = df[(df.loc[:,'month'].values ==month_from)]
    df3 = df2[(df2.loc[:,'year'].values == year_from)]
    final = df3[(df3.loc[:,'hour'].values == hour_from)]
    x = final.iloc[:,:-1].values
    result = loaded_model.predict(x)
    print(result)
    return result
    