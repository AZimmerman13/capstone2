import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import importlib
matplotlib.use("Agg")
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV, Ridge, LinearRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline as SKPipe
from sklearn.inspection import permutation_importance, plot_partial_dependence
from sklearn.model_selection import GridSearchCV
from src.pipeline import Pipeline
from src.main_script import plot_corr_matrix, scree_plot, plot_num_estimators_mse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor


if __name__ == '__main__':
    print("Loading Data")
    # read in files from s3 bucket
    energy = Pipeline('s3://ajzcap2/energy_dataset.csv')
    weather = Pipeline('s3://ajzcap2/weather_features.csv')

    #make index a datetime object
    energy.my_reset_index()
    weather.my_reset_index()

    # Clean Catagoricals
    weather.clean_categoricals(['weather_main'])

    # Drop columns
    weather_drop_cols = ['weather_icon', 'weather_description', 'weather_id', 'temp_min', 
                    'temp_max', 'pressure', 'humidity',
                    'rain_1h', 'rain_3h', 'snow_3h', 'clouds_all', 'dust', 'fog', 'haze',
                     'mist', 'rain', 'smoke', 'snow', 'squall', 'thunderstorm', 'clouds', 'drizzle', 'wind_deg']
    
    energy_drop_cols = ['generation fossil coal-derived gas','generation fossil oil shale', 
                        'generation fossil peat', 'generation geothermal',
                        'generation marine', 'generation hydro pumped storage aggregated',
                         'forecast wind offshore eday ahead', 'generation wind offshore', 
                         'price day ahead', 'total load forecast', 'forecast wind onshore day ahead', 
                         'forecast solar day ahead']

    for i in weather_drop_cols:
        weather.df.drop(i, axis=1, inplace=True)
    for i in energy_drop_cols:
        energy.df.drop(i, axis=1, inplace=True)

    # propagate last valid observation forward to next valid to fill NaNs
    for i in energy.df.columns:
        energy.df[i].fillna(method='pad', inplace=True)

    

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

    # clean residual col names that came from the merge and low feature importance features
    for i in ["Valencia_city_name", " Barcelona_city_name", "Bilbao_city_name", 
            "Seville_city_name", "Madrid_city_name", 'Seville_wind_speed',
             " Barcelona_wind_speed", 'Valencia_wind_speed']:
        all_cities_df.df.drop(i, axis=1, inplace=True)

    
    # Transformations
    print('\nPerforming transformations')
  

    # Merge energy with the featurized cities DF to make the complete DataFrame
    full_df = energy.merge_dfs(all_cities_df.df)

    plot_corr_matrix(full_df.df)
    plt.savefig('images/full_corr_sparse.png')
    plt.close()

    # get_top_abs_correlations(full_df.df, 10)


    print('\nCreating train, test, and holdout sets')
    full_df.getXy('price actual')
    # full_df.create_holdout()

    plot_corr_matrix(energy.df)
    plt.savefig('images/clean_energy_corr_sparse.png')
    # plt.show()
    plt.close()
    plot_corr_matrix(pd.concat([all_cities_df.df, full_df.y], axis=1))
    plt.savefig('images/clean_weather_corr_sparse.png')
    # plt.show()
    plt.close()

    X = full_df.X
    y = full_df.y

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print('\n trying a few models')
    # models = [RandomForestRegressor(n_estimators=20, n_jobs=-1, max_features='sqrt'), Lasso(alpha=0.03), Ridge(alpha=0.03), LinearRegression(n_jobs=-1)]

    # for model in models:
    #     pipe = SKPipe([('scaler', StandardScaler()), (f'{model}', model)], verbose=True)
    #     pipe.fit(X_train, y_train)
        
    #     train_score = pipe.score(X_train, y_train)
    #     test_score = pipe.score(X_test, y_test)
        
    #     print(f"{model} \n\ntest score = {test_score}\n train score = {train_score}\n\n")

    # # PCA
    # print("\nLet's try PCA")
    # pca = PCA(n_components=50)
    # X_pca = pca.fit_transform(full_df.X_std)

    # fig, ax = plt.subplots(figsize=(10, 6))
    # scree_plot(ax, pca, title="Scree Plot for Energy Principal Components")
    # plt.savefig('images/pca_full_sparse.png')
    # plt.close()
    
    print('Gridsearch time, go get some coffee')
    # parameters = {'n_estimators': (2, 5, 10, 20, 30), 
    #             'max_depth': (None, 5, 7), 
    #             'max_features': ('auto', 'sqrt', 'log2')}
    # rf = RandomForestRegressor(verbose=True, n_jobs=-1)
    # grid = GridSearchCV(rf, parameters, verbose=1, n_jobs=-1)

    # grid.fit(X_train,y_train)
    # gridscore_test = grid.score(X_test, y_test)
    # grisdcore_train = grid.score(X_train, y_train)

    # grid.best_params_
    # Out[4]: {'max_depth': None, 'max_features': 'auto', 'n_estimators': 30}

    rf = RandomForestRegressor(max_depth=None, max_features='auto', n_estimators=30)

    feature_names = full_df.X.columns
    rf.fit(X_train, y_train)

    feat_imp = pd.DataFrame({'feature_name':feature_names, 'feat_imp': rf.feature_importances_})
    feat_imp.sort_values('feat_imp',ascending=False,inplace=True)
    fig, ax = plt.subplots(1, figsize=(8,10))
    ax.barh(feat_imp['feature_name'], feat_imp['feat_imp'])
    ax.invert_yaxis()
    ax.set_title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.savefig('images/feature_imp_sparse.png')
    plt.close()


    plot_partial_dependence(rf, X_train, feature_names, feature_names=feature_names)
    fig = plt.gcf()
    fig.suptitle("Partial Dependence of energy price")
    plt.savefig('images/partial_dependence_sparse.png')
    plt.close()




    # print(f"\n\n\nR2 test with best params = {gridscore_test}")
    # print(f"\n\nR2 train with best params = {grisdcore_train}")