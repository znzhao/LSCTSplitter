import pandas as pd
def load(path):
    '''
    Load the yield curve data from the saved drive
    '''
    data = pd.read_csv(path, index_col = 'date')
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    return data