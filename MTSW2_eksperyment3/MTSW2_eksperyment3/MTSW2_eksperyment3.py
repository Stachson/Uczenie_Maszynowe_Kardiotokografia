import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from scipy.stats import ttest_rel
from tabulate import tabulate

# zast¹pienie wartoœci nan median¹ cechy
def nan_to_median(x):
    for col in range(x.shape[1]):
        nan_indices = np.isnan(x[:, col])
        if np.any(nan_indices):
            median_value = np.nanmedian(x[:, col])
            x[nan_indices, col] = median_value
    return x

#usuwanie rekordów z nan
def nan_delete_x(x):
    nan_indices = np.any(np.isnan(x), axis=1)
    return nan_indices

def mean_imputation(x):
    imputer = SimpleImputer(strategy='mean')
    x_imputed = imputer.fit_transform(x)
    return x_imputed

def nearest_neighbors_imputation(x):
    imputer = KNNImputer(n_neighbors=5)
    x_imputed = imputer.fit_transform(x)
    return x_imputed

def run_experiment_3(x, y, clf, feature_selection_method, n_splits=5, n_repeats=2, random_state=42, max_features=20, top_n_features=20):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    scores = np.zeros((max_features, n_splits * n_repeats))

    for num_features in range(1, max_features + 1):
        for fold_id, (train, test) in enumerate(rskf.split(x, y)):
            # Oblicz macierz korelacji Pearsona
            #corr_matrix = np.corrcoef(x[train], rowvar=False)

            #selected_features = np.abs(corr_matrix[-1, :-1])
            #top_features_indices = np.argsort(selected_features)[::-1]

            model = RFE(clf, n_features_to_select=num_features)
            model.fit(x[train], y[train])
            selected_indices = np.where(model.support_)[0]

            # Wybierz top N cech
            #selected_indices = top_features_indices[:num_features]

            # Dopasuj model do zredukowanego zbioru cech
            clf.fit(x[train][:, selected_indices], y[train])

            y_pred = clf.predict(x[test][:, selected_indices])

            scores[num_features - 1, fold_id] = accuracy_score(y[test], y_pred)

    return scores


#wczytanie danych
dataset = np.genfromtxt("ctg_10.csv", delimiter=",")
x = dataset[:, :-1]
y = dataset[:, -1].astype(int)


#tablica modeli klasyfikacyjnych
clfs = {
'GNB': GaussianNB(),
'kNN': KNeighborsClassifier(),
'CART': DecisionTreeClassifier(random_state=42),
}

imputation_methods = ['median', 'mean', 'nearest_neighbors']

feature_selection_methods = ['pearson', 'select_from_model', 'rfe']

x = mean_imputation(x) # imputacja dla pozosta³ych eksperymentów

# Eksperyment 3
print("\nEksperyment 3 (CART with RFE feature selection):")
clf_knn = clfs['CART']
max_features = 20

scores_exp3 = run_experiment_3(x, y, clf_knn, 'rfe', max_features=max_features)
mean_exp3 = np.mean(scores_exp3, axis=1)
std_exp3 = np.std(scores_exp3, axis=1)

for num_features in range(1, max_features + 1):
    print("Num Features = %d: %.3f (%.2f)" % (num_features, mean_exp3[num_features - 1], std_exp3[num_features - 1]))
