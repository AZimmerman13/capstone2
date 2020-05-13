import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import importlib
matplotlib.use("Agg")
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Lasso, LassoCV
from src.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


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
    # plt.tight_layout()

def get_redundant_pairs(df):
     '''Get diagonal and lower triangular pairs of correlation matrix'''
     pairs_to_drop = set()
     cols = df.columns
     for i in range(0, df.shape[1]):
         for j in range(0, i+1):
             pairs_to_drop.add((cols[i], cols[j]))
     return pairs_to_drop

def get_top_abs_correlations(df, n=5):
     au_corr = df.corr().abs().unstack()
     labels_to_drop = get_redundant_pairs(df)
     au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
     return au_corr[0:n]



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
    weather_drop_cols = ['weather_icon', 'weather_main', 'weather_id', 'temp_min', 'temp_max']
    energy_drop_cols = ['generation fossil coal-derived gas','generation fossil oil shale', 
                        'generation fossil peat', 'generation geothermal',
                        'generation marine', 'generation hydro pumped storage aggregated',
                         'forecast wind offshore eday ahead', 'generation wind offshore', 
                         'price day ahead', 'total load forecast', 'forecast wind onshore day ahead', 'forecast solar day ahead']

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

    get_top_abs_correlations(full_df.df, 10)


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

    print("Lasso time: yee-haw")
   
    

# LassoCV

    X = full_df.X_std
    y = full_df.y_train

    reg = LassoCV(random_state=0, verbose=True, n_jobs=-1, max_iter = 10000)
    reg.fit(X,y)
    regscore = reg.score(X, y)
    print(regscore)
    y_preds = reg.predict(full_df.Xscaler.transform(full_df.X_test))
    best_alpha = reg.alpha_
    coefs = reg.coef_
    print(f"Best alpha = {best_alpha}")
    

# VIF
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["features"] = full_df.X.columns

    print(vif.sort_values('VIF Factor', ascending=False).head(20).round(1))

    print('all done.')


    
