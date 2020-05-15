import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import importlib
matplotlib.use("Agg")
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV
from sklearn.decomposition import PCA
from src.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor




def plot_corr_matrix(df):
    f = plt.figure(figsize=(19, 15))
    plt.matshow(df.corr(), fignum=f.number)
    plt.xticks(range(df.shape[1]), df.columns, fontsize=16, rotation=90)
    plt.yticks(range(df.shape[1]), df.columns, fontsize=16)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.xlabel('Correlation Matrix', fontsize=22)
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

def scree_plot(ax, pca, n_components_to_plot=8, title=None): # Credit: Galvanize Data Science
    """Make a scree plot showing the variance explained (i.e. varaince of the projections) for the principal components in a fit sklearn PCA object.
    
    Parameters
    ----------
    ax: matplotlib.axis object
      The axis to make the scree plot on.
      
    pca: sklearn.decomposition.PCA object.
      A fit PCA object.
      
    n_components_to_plot: int
      The number of principal components to display in the skree plot.
      
    title: str
      A title for the skree plot.
    """
    num_components = pca.n_components_
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
    ax.plot(ind, vals, color='blue')
    ax.scatter(ind, vals, color='blue', s=50)

    for i in range(num_components):
        ax.annotate(r"{:2.2f}".format(vals[i]), 
                   (ind[i]+0.2, vals[i]+0.005), 
                   va="bottom", 
                   ha="center", 
                   fontsize=12)

    ax.set_xticklabels(ind, fontsize=12)
    ax.set_ylim(0, max(vals) + 0.05)
    ax.set_xlim(0 - 0.45, n_components_to_plot + 0.45)
    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)
    if title is not None:
        ax.set_title(title, fontsize=16)


def plot_num_estimators_mse(num_estimator_list, train_errors_rf, test_errors_rf):
    plt.figure(figsize=(15,10))
    plt.plot(num_estimator_list, train_errors_rf, label='Training MSE')
    plt.plot(num_estimator_list, test_errors_rf, label='Test MSE')
    plt.xlabel('Number of Estimators')
    plt.ylabel('MSE')
    plt.xscale('linear')
    plt.title('Random Forest MSE vs. Num Estimators')
    plt.legend()
    

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

if __name__ == '__main__':
    pass
   