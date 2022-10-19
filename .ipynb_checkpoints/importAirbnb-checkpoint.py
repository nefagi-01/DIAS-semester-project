import pandas as pd

def importAirbnb(path):
    df = pd.read_csv(path)
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    newdf = df.select_dtypes(include=numerics).dropna(axis=1).drop(columns=['id', 'host_id'])
    return newdf.to_numpy()