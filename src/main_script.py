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
    # read in files from s3 bucket
    energy = Pipeline('s3://ajzcap2/energy_dataset.csv')
    weather = Pipeline('s3://ajzcap2/weather_features.csv')

    #make index a datetime object
    energy.reset_index()
    weather.reset_index()
    
    print('\nPerforming transformations')
    merged_df = energy.merge_dfs(weather.df)
    weather.consolidate('dt_iso')
    merged_by_date = energy.merge_dfs(weather.grouped_avg)

    print('Creating train, test, and holdout sets')
    merged_by_date.getXy('price actual')
    merged_by_date.create_holdout()

    print('\nWriting train, test, and holdouts to filesystem')

    train_test_split_holdout_list = [merged_by_date.X_train, merged_by_date.X_test, 
                                    merged_by_date.X_holdout, merged_by_date.y_train, 
                                    merged_by_date.y_test, merged_by_date.y_holdout]

    ttsh_filenames = ['X_train', 'X_test', 'X_holdout', 'y_train', 
                    'y_test', 'y_holdout']

    for (i, fname) in zip(train_test_split_holdout_list, ttsh_filenames):
            i.to_csv(f"data/{fname}.csv")
   
    for (i, fname) in zip(train_test_split_holdout_list, ttsh_filenames):
        i.to_csv(f's3://ajzcap2/{fname}.csv')
   

    
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


    