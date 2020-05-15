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

# plt.style.use('fivethirtyeight')


def gridsearch():
    parameters = {'n_estimators': (2, 5, 10, 20, 30), 
                'max_depth': (None, 5, 7), 
                'max_features': ('auto', 'sqrt', 'log2')}
    rf = RandomForestRegressor(verbose=True, n_jobs=-1)
    grid = GridSearchCV(rf, parameters, verbose=1, n_jobs=-1)

    grid.fit(X_train,y_train)
    gridscore_test = grid.score(X_test, y_test)
    grisdcore_train = grid.score(X_train, y_train)
    return gridscore_train, gridscore_test, grid.best_params_

def pdplots():

    first_pdp = ['generation fossil gas', 
            'generation fossil hard coal', 'total load actual'] 
             
    second_pdp = ['generation other renewable',
             'generation solar']

    third_pdp = ['generation wind onshore', 'generation nuclear', 
             'Madrid_wind_speed']

    fourth_pdp = ['generation hydro pumped storage consumption']

    plot_partial_dependence(rf, X_train, first_pdp, n_jobs=-1)
    fig.suptitle("Partial Dependence of Energy Price on Various Generation Types")
    fig.subplots_adjust(hspace=2.0, wspace=2.0)
    plt.tight_layout()
    plt.savefig('images/pd1.png')
    plt.close()

    plot_partial_dependence(rf, X_train, second_pdp, n_jobs=-1)
    fig.suptitle("Partial Dependence of Energy Price on Various Generation Types")
    fig.subplots_adjust(hspace=2.0, wspace=2.0)
    plt.tight_layout()
    plt.savefig('images/pd2.png')
    plt.close()

    plot_partial_dependence(rf, X_train, third_pdp, n_jobs=-1)
    fig.suptitle("Partial Dependence of Energy Price on Various Generation Types")
    fig.subplots_adjust(hspace=2.0, wspace=2.0)
    plt.tight_layout()
    plt.savefig('images/pd3.png')
    plt.close()

    plot_partial_dependence(rf, X_train, fourth_pdp, n_jobs=-1)
    fig.suptitle("Partial Dependence of Energy Price on Various Generation Types")
    fig.subplots_adjust(hspace=2.0, wspace=2.0)
    plt.tight_layout()
    plt.savefig('images/pd4.png')
    plt.close()

def compare_default_models():
    models = [RandomForestRegressor(n_jobs=-1), Lasso(), Ridge(), LinearRegression(n_jobs=-1)]

    for model in models:
        pipe = SKPipe([('scaler', StandardScaler()), (f'{model}', model)], verbose=True)
        pipe.fit(X_train, y_train)
        
        train_score = pipe.score(X_train, y_train)
        test_score = pipe.score(X_test, y_test)
        
        print(f"{model} \n\ntest score = {test_score}\n train score = {train_score}\n\n")

def pca_with_scree():
    print("\nLet's try PCA")
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(full_df.X_std)

    fig, ax = plt.subplots(figsize=(10, 6))
    scree_plot(ax, pca, title="Scree Plot for Energy Principal Components")
    plt.savefig('images/pca_full_sparse.png')
    plt.close()

def feat_imp_plots():
    feature_names = full_df.X.columns
    feat_imp = pd.DataFrame({'feature_name':feature_names, 'feat_imp': rf.feature_importances_})
    feat_imp.sort_values('feat_imp',ascending=False,inplace=True)
    fig, ax = plt.subplots(1, figsize=(8,10))
    ax.barh(feat_imp['feature_name'].head(9), feat_imp['feat_imp'].head(9))
    ax.invert_yaxis()
    ax.set_title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.savefig('images/feature_imp_sparse.png')
    plt.close()

def plot_oob_error():
    fig, ax = plt.subplots()
    oob_diff = []
    oob = []
    for i in list(range(2,20)):
        rf = RandomForestRegressor(max_depth=i, max_features='auto', n_estimators=30, oob_score=True, n_jobs=-1)

        rf.fit(X_train, y_train)

        print(f"R2 Train = {rf.score(X_train, y_train)}")
        print(f"R2 Test = {rf.score(X_test, y_test)}")
        print(f"R2 Holdout = {rf.score(X_holdout, y_holdout)}")
        print(f"OOB score = {rf.oob_score_}")
        oob_diff.append(rf.score(X_train, y_train) - rf.oob_score_)
        oob.append(rf.oob_score_)

    ax.plot(oob_diff, color='red')
    ax.plot(oob, color='blue')
    ax.set_title("Reducing OOB Error by limiting max_depth")
    plt.savefig('images/oob.png')

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
                    'temp_max', 'pressure', 'humidity','rain_1h', 'rain_3h', 'snow_3h',
                     'clouds_all', 'dust', 'fog', 'haze','mist', 'rain', 'smoke', 
                     'snow', 'squall', 'thunderstorm', 'clouds', 'drizzle', 'wind_deg']
    
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
             " Barcelona_wind_speed", "Valencia_temp"]:
        all_cities_df.df.drop(i, axis=1, inplace=True)

    # Merge energy and weather
    print('\nMerging dataset')
  
    # Merge energy with the featurized cities DF to make the complete DataFrame
    full_df = energy.merge_dfs(all_cities_df.df)

    plot_corr_matrix(full_df.df)
    plt.savefig('images/full_corr_sparse.png')
    plt.close()

    print('\nCreating train, test, and holdout sets')
    full_df.getXy('price actual')
    full_df.create_holdout()

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

    # X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    X_train = full_df.X_train
    y_train = full_df.y_train
    X_test = full_df.X_test
    y_test = full_df.y_test
    X_holdout = full_df.X_holdout
    y_holdout = full_df.y_holdout



    print('\ntrying a few models')
    # compare_default_models()

    # PCA
    # pca_with_scree()
    
    print('Gridsearch time, go get some coffee')
    # gridsearch()

    # grid.best_params_
    # Out[4]: {'max_depth': None, 'max_features': 'auto', 'n_estimators': 30}
    

    #Best Model
    rf = RandomForestRegressor(max_depth=None, max_features='auto', n_estimators=30, oob_score=True, n_jobs=-1)
    rf.fit(X_train, y_train)

    # plot_oob_error()
    
    # Check feature importances
    feat_imp_plots()

    # Check partial dependences
    # pdplots()
