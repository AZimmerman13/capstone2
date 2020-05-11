import pandas as pd
import numpy as np
import matplotlib
# matplotlib.use("Agg")
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





    # Drop columns
    weather_drop_cols = ['weather_icon', 'weather_main', 'weather_id']
    energy_drop_cols = ['generation fossil coal-derived gas','generation fossil oil shale', 
                        'generation fossil peat', 'generation geothermal',
                        'generation marine', 'generation hydro pumped storage aggregated']

    for i in weather_drop_cols:
        weather.df.drop(i, axis=1, inplace=True)
    for i in energy_drop_cols:
        energy.df.drop(i, axis=1, inplace=True)

    # Demonstrate over-featurization of weather.df
    # for i in weather.df.weather_description.unique():
    #     print(f"{i} = {weather.df.weather_id[weather.df.weather_description == i].unique()}, {weather.df.weather_main[weather.df.weather_description == i].unique()}")




    # plot_corr_matrix(energy.df)
    # plt.savefig('images/energy_corr.png')
    # # plt.show()
    # plt.close()
    # plot_corr_matrix(weather.df)
    # plt.savefig('images/weather_corr.png')
    # # plt.show()
    # plt.close()


    city_df_list = weather.featurize_cities(['Valencia', 'Madrid', "Bilbao", 'Barcelona', 'Sevilla'])

    valencia = Pipeline.from_df(city_df_list[0])
    madrid = Pipeline.from_df(city_df_list[1])
    bilbao = Pipeline.from_df(city_df_list[2])
    barcelona = Pipeline.from_df(city_df_list[3])
    sevilla = Pipeline.from_df(city_df_list[4])

    valencia.merge_dfs(madrid)


   





    
    # Transformations
    print('\nPerforming transformations')
    weather_cols = ['weather_description', 'weather_main', 'weather_id']
    # weather.clean_categoricals(weather_cols)
    energy_cols = []

    # weather.featurize_col("city_name", weather.df.city_name.unique())

    merged_df = energy.merge_dfs(weather.df)
    weather.consolidate('dt_iso')
    merged_by_date = energy.merge_dfs(weather.grouped_avg)

    # weather.df.set_index('city_name', append=True, inplace=False, drop=False)

    # test = test.reset_index()

    # test.set_index(['dt_iso'])
    # weather.df.set_index('city_name', append=True)

# for feat in ["Madrid", "Valencia"]:
#     for col in weather.df.columns:
#         weather.df[f"{feat}_{col}"] = weather.df[col][weather.df['city_name'] == "Madrid"]

# cols = test.columns
# for feat in ["Madrid", "Valencia"]:
#     for col in cols[1:]:
#         test[f"{feat}_{col}"] = test[col][test['city_name'] == feat]






    print('\nCreating train, test, and holdout sets')
    merged_by_date.getXy('price actual')
    merged_by_date.create_holdout()

    # print('\nWriting train, test, and holdouts to filesystem')

    train_test_split_holdout_list = [merged_by_date.X_train, merged_by_date.X_test, 
                                    merged_by_date.X_holdout, merged_by_date.y_train, 
                                    merged_by_date.y_test, merged_by_date.y_holdout]

    ttsh_filenames = ['X_train', 'X_test', 'X_holdout', 'y_train', 
                    'y_test', 'y_holdout']


    # Dont need to to this everytime I run the script for EDA
    # for (i, fname) in zip(train_test_split_holdout_list, ttsh_filenames):
    #         i.to_csv(f"data/{fname}.csv")
   
    # for (i, fname) in zip(train_test_split_holdout_list, ttsh_filenames):
    #     i.to_csv(f's3://ajzcap2/{fname}.csv')
   

    

  

    
    plot_corr_matrix(energy.df)
    plt.savefig('images/clean_energy_corr.png')
    # plt.show()
    plt.close()
    plot_corr_matrix(weather.df)
    plt.savefig('images/clean_weather_corr.png')
    # plt.show()
    plt.close()
    
    print('all done.')


    