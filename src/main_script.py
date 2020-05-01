import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import importlib
from src.pipeline import Pipeline
# importlib.reload(src.pipeline)
# from src.pipeline import Pipeline



def plot_corr_matrix(df):
    f = plt.figure(figsize=(19, 15))
    plt.matshow(df.corr(), fignum=f.number)
    plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
    plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)





if __name__ == '__main__':
    # energy_df = pd.read_csv('data/energy_dataset.csv',index_col=0, parse_dates=[0])
    # weather_df = pd.read_csv('data/weather_features.csv',index_col=0, parse_dates=[0])
    print("Loading Data")
    energy = Pipeline('data/energy_dataset.csv')
    weather = Pipeline('data/weather_features.csv')
    # energy_df.index = pd.to_datetime(energy_df.index, utc=True)
    # weather_df.index = pd.to_datetime(weather_df.index, utc=True)
    
    merged_df = energy.merge_dfs(weather.df)
    weather.consolidate('dt_iso')
    merged_by_date = energy.merge_dfs(weather.grouped_avg)
    merged_by_date.getXy('price actual')
    merged_by_date.create_holdout()
    
    # energy_df.merge(weather_df, right_index=True, left_index=True)  

    # cities_gb = weather_df.groupby("dt_iso")
    # cities_avg = cities_gb.mean()

    # plt.matshow(energy_df.corr())
    # plt.show()
    # print('creating matrices')
    # plot_corr_matrix(energy_df)
    # plt.savefig('images/energy_corr.png')
    # plt.close()
    # plot_corr_matrix(weather_df)
    # plt.savefig('images/weather_corr.png')
    # plt.close()
    print('all done.')


    