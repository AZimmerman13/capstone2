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
    plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=90)
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
    energy.my_reset_index()
    weather.my_reset_index()

    # Drop columns
    weather_drop_cols = ['weather_icon', 'weather_main', 'weather_id']
    energy_drop_cols = ['generation fossil coal-derived gas','generation fossil oil shale', 
                        'generation fossil peat', 'generation geothermal',
                        'generation marine', 'generation hydro pumped storage aggregated',
                         'forecast wind offshore eday ahead', 'generation wind offshore', 
                         'price day ahead', 'total load forecast']

    for i in weather_drop_cols:
        weather.df.drop(i, axis=1, inplace=True)
    for i in energy_drop_cols:
        energy.df.drop(i, axis=1, inplace=True)

    # propagate last valid observation forward to next valid to fill NaNs
    for i in energy.df.columns:
        energy.df[i].fillna(method='pad', inplace=True)

    # for i in energy.df.columns:
    #     print(f"{i}: missing {energy.df[i].isna().sum()}")

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
    # Clean Catagoricals
    weather.clean_categoricals(['weather_description'])



    #Featurizing Cities
    city_df_list = weather.featurize_cities(['Valencia', 'Madrid', "Bilbao", ' Barcelona', 'Seville'])

    valencia = Pipeline.from_df(city_df_list[0])
    madrid = Pipeline.from_df(city_df_list[1])
    bilbao = Pipeline.from_df(city_df_list[2])
    barcelona = Pipeline.from_df(city_df_list[3])
    sevilla = Pipeline.from_df(city_df_list[4])

    # There has GOT to be a better way to do this
    vm = valencia.merge_dfs(madrid.df)
    bb = bilbao.merge_dfs(barcelona.df)
    sbb = sevilla.merge_dfs(bb.df)
    all_cities_df = vm.merge_dfs(sbb.df)

    # clean residual col names that came from the merge
    for i in ["Valencia_city_name", " Barcelona_city_name", "Bilbao_city_name", 
            "Seville_city_name", "Madrid_city_name", 'Seville_snow_3h', ' Barcelona_snow_3h']:
        all_cities_df.df.drop(i, axis=1, inplace=True)

    
    # Transformations
    print('\nPerforming transformations')
    weather_cols = ['weather_description', 'weather_main', 'weather_id']
    # weather.clean_categoricals(weather_cols)
    energy_cols = []

    # Merge energy with the featurized cities DF to make the complete DataFrame
    full_df = energy.merge_dfs(all_cities_df.df)

    plot_corr_matrix(full_df.df)
    plt.savefig('images/full_corr.png')
    plt.close()


    print('\nCreating train, test, and holdout sets')
    full_df.getXy('price actual')
    full_df.create_holdout()

    print('\nWriting train, test, and holdouts to filesystem')

    train_test_split_holdout_list = [full_df.X_train, full_df.X_test, 
                                    full_df.X_holdout, full_df.X_std, full_df.y_train, 
                                    full_df.y_test, full_df.y_holdout]

    ttsh_filenames = ['X_train', 'X_test', 'X_holdout', 'X_std','y_train', 
                    'y_test', 'y_holdout']


    # Dont need to to this everytime I run the script for EDA
    # for (i, fname) in zip(train_test_split_holdout_list, ttsh_filenames):
    #         i.to_csv(f"data/{fname}.csv")
   
    # for (i, fname) in zip(train_test_split_holdout_list, ttsh_filenames):
    #     if type(i) == 'numpy.ndarray':
    #         i.to_csv(f's3://ajzcap2/{fname}.csv')
   

    

  

    
    plot_corr_matrix(energy.df)
    plt.savefig('images/clean_energy_corr.png')
    # plt.show()
    plt.close()
    plot_corr_matrix(weather.df)
    plt.savefig('images/clean_weather_corr.png')
    # plt.show()
    plt.close()

    
    
    print('all done.')


    
