import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import importlib
matplotlib.use("Agg")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from src.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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

def train_at_various_alphas(X, y, model, alphas, n_folds=10, **kwargs): # Credit: Gavanize Data Science
    """Train a regularized regression model using cross validation at various values of alpha.
    
    Parameters
    ----------
    
    X: np.array
      Matrix of predictors.
      
    y: np.array
      Target array.
      
    model: sklearn model class
      A class in sklearn that can be used to create a regularized regression object.  Options are `Ridge` and `Lasso`.
      
    alphas: numpy array
      An array of regularization parameters.
      
    n_folds: int
      Number of cross validation folds.
      
    Returns
    -------
    
    cv_errors_train, cv_errors_test: tuple of DataFrame
      DataFrames containing the training and testing errors for each value of 
      alpha and each cross validation fold.  Each row represents a CV fold,
      and each column a value of alpha.
    """
    cv_errors_train = pd.DataFrame(np.empty(shape=(n_folds, len(alphas))),
                                     columns=alphas)
    cv_errors_test = pd.DataFrame(np.empty(shape=(n_folds, len(alphas))),
                                        columns=alphas)
    for alpha in alphas:
        train_fold_errors, test_fold_errors = cv(X, y, model(alpha=alpha, **kwargs), n_folds=n_folds)
        cv_errors_train.loc[:, alpha] = train_fold_errors
        cv_errors_test.loc[:, alpha] = test_fold_errors
    return cv_errors_train, cv_errors_test

def get_optimal_alpha(mean_cv_errors_test): # Credit: Galvanize Data Science
    alphas = mean_cv_errors_test.index
    optimal_idx = np.argmin(mean_cv_errors_test.values)
    optimal_alpha = alphas[optimal_idx]
    return optimal_alpha



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
    model = Lasso(max_iter=1000, alpha=1.0)
    model.fit(full_df.X_std, full_df.y_train)
    y_pred = model.predict(full_df.Xscaler.transform(full_df.X_test))

    lasso_alphas = np.logspace(-2, 4, num=300)

    lasso_cv_errors_train, lasso_cv_errors_test = train_at_various_alphas(
    full_df.X_std.values, full_df.y_train.values, Lasso, lasso_alphas)

    lasso_mean_cv_errors_train = lasso_cv_errors_train.mean(axis=0)
    lasso_mean_cv_errors_test = lasso_cv_errors_test.mean(axis=0)

    lasso_optimal_alpha = get_optimal_alpha(lasso_mean_cv_errors_test)
        
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(np.log10(lasso_alphas), lasso_mean_cv_errors_train)
    ax.plot(np.log10(lasso_alphas), lasso_mean_cv_errors_test)
    ax.axvline(np.log10(lasso_optimal_alpha), color='grey')
    ax.set_title("LASSO Regression Train and Test MSE")
    ax.set_xlabel(r"$\log(\alpha)$")
    ax.set_ylabel("MSE")
    plt.savefig('images/lasso_errors_vs_alpha.png')
    plt.close()

    lasso_models = []

    for alpha in lasso_alphas:
        scaler = StandardScaler()
        scaler.fit(full_df.X_train.values, full_df.y_train.values)
        X_train_std, y_train_std = scaler.transform(full_df.X_train.values, full_df.y_train.values)
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_train_std, y_train_std)
        lasso_models.append(lasso)

    paths = pd.DataFrame(np.empty(shape=(len(lasso_alphas), len(X_train.columns))),
                     index=lasso_alphas, columns=X_train.columns)

    for idx, model in enumerate(lasso_models):
        paths.iloc[idx] = model.coef_
        
    fig, ax = plt.subplots(figsize=(14, 4))
    for column in full_df.X_train.columns:
        path = paths.loc[:, column]
        ax.plot(np.log10(lasso_alphas), path, label=column)
    ax.axvline(np.log10(lasso_optimal_alpha), color='grey')
    ax.legend(loc='lower right')
    ax.set_title("LASSO Regression, Standardized Coefficient Paths")
    ax.set_xlabel(r"$\log(\alpha)$")
    ax.set_ylabel("Standardized Coefficient")

    plt.savefig('images/lasso_coeff_paths.png')




    print('all done.')


    
