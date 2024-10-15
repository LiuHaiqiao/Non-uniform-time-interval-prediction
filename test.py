import pandas as pd
from utils import date_to_tensor
from Embed import TemporalEmbedding

def read_csv(path):
    data = pd.read_csv(path)
    print(data.shape)

def date_embedding():
    dates = '2021/08/12'
    tensor = date_to_tensor(dates)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    embedding = TemporalEmbedding(d_model=128)
    output = embedding(tensor)
    print(output.shape)
    print(output)




if __name__ == '__main__':
    date_embedding()