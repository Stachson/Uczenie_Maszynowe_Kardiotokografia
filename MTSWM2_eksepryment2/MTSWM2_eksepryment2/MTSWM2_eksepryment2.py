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

def run_experiment_2(x, y, clf, feature_selection_method, n_splits=5, n_repeats=2, random_state=42, top_n_features=20):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    scores = np.zeros(n_splits * n_repeats)

    for fold_id, (train, test) in enumerate(rskf.split(x, y)):
        # Wybór metody selekcji cech
        if feature_selection_method == 'pearson':
            corr_matrix = np.corrcoef(x[train], rowvar=False)
            selected_features = np.abs(corr_matrix[-1, :-1])
            top_features_indices = np.argsort(selected_features)[::-1]
            selected_indices = top_features_indices[:top_n_features]
        elif feature_selection_method == 'select_from_model':
            model = SelectFromModel(clf)
            model.fit(x[train], y[train])
            selected_indices = model.get_support(indices=True)
        elif feature_selection_method == 'rfe':
            model = RFE(clf, n_features_to_select=top_n_features)
            model.fit(x[train], y[train])
            selected_indices = np.where(model.support_)[0]
        
        # Dopasuj model do zredukowanego zbioru cech
        clf.fit(x[train][:, selected_indices], y[train])

        y_pred = clf.predict(x[test][:, selected_indices])

        scores[fold_id] = accuracy_score(y[test], y_pred)

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

# Eksperyment 2

print("\nEksperyment 2 (CART with feature selection methods):")
clf_cart = clfs['CART']
all_scores_exp2 = np.empty((0, 10))
for method in feature_selection_methods:
    scores_exp2 = run_experiment_2(x, y, clf_cart, method)
    all_scores_exp2 = np.vstack((all_scores_exp2, scores_exp2))
    mean_exp2 = np.mean(scores_exp2)
    std_exp2 = np.std(scores_exp2)
    print("%s: %.3f (%.2f)" % (method, mean_exp2, std_exp2))

np.save('results_exp2', all_scores_exp2)

scores = np.load('results_exp2.npy')

alfa = .05
t_statistic = np.zeros((len(clfs), len(clfs)))
p_value = np.zeros((len(clfs), len(clfs)))

for i in range(len(clfs)):
    for j in range(len(clfs)):
        t_statistic[i, j], p_value[i, j] = ttest_rel(scores[i], scores[j])
print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)

print("\n\n")

headers = ["CART P", "CART SFM", "CART RFE"]
names_column = np.array([["CART P"], ["CART SFM"], ["CART RFE"]])
t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)


advantage = np.zeros((len(clfs), len(clfs)))
advantage[t_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
(names_column, advantage), axis=1), headers)
print("Advantage:\n", advantage_table)

significance = np.zeros((len(clfs), len(clfs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
(names_column, significance), axis=1), headers)
print("Statistical significance (alpha = 0.05):\n", significance_table)

stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate(
(names_column, stat_better), axis=1), headers)
print("Statistically significantly better:\n", stat_better_table)
