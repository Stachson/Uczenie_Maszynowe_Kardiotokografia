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

#usuwanie rekordów z nan    nieu¿ywana
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


#3 modele klasyfikacyjne, wspó³czynnik korelacji Pearsona dla 20 cech, zast¹pienie brakuj¹cych wartoœci ró¿nymi sposobami
def run_experiment_1(x, y, clfs, imputation_methods, n_splits=5, n_repeats=2, random_state=42, top_n_features=20):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    scores = np.zeros((len(clfs), len(imputation_methods), n_splits * n_repeats))

    for fold_id, (train, test) in enumerate(rskf.split(x, y)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])

            for imputation_method_id, imputation_method in enumerate(imputation_methods):
                # Kopiujemy oryginalny zbiór danych przed zastosowaniem imputacji
                x_train, x_test = np.copy(x[train]), np.copy(x[test])

                # Zastosowanie wybranej metody imputacji dla zbioru treningowego
                if imputation_method == 'median':
                    x_train = nan_to_median(x_train)
                elif imputation_method == 'mean':
                    x_train = mean_imputation(x_train)
                elif imputation_method == 'nearest_neighbors':
                    x_train = nearest_neighbors_imputation(x_train)

                # Zastosowanie wybranej metody imputacji dla zbioru testowego
                if imputation_method == 'median':
                    x_test = nan_to_median(x_test)
                elif imputation_method == 'mean':
                    x_test = mean_imputation(x_test)
                elif imputation_method == 'nearest_neighbors':
                    x_test = nearest_neighbors_imputation(x_test)

                # oblicz macierz korelacji Pearsona
                corr_matrix = np.corrcoef(x_train, rowvar=False)
                selected_features = np.abs(corr_matrix[-1, :-1])
                top_features_indices = np.argsort(selected_features)[::-1]
                selected_indices = top_features_indices[:top_n_features]

                # Dopasuj model do zbioru treningowego z zastosowan¹ imputacj¹
                clf.fit(x_train[:, selected_indices], y[train])

                y_pred = clf.predict(x_test[:, selected_indices])

                scores[clf_id, imputation_method_id, fold_id] = accuracy_score(y[test], y_pred)


    return scores

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

# Eksperyment 1
print("Eksperyment 1:")
scores_exp1 = run_experiment_1(x, y, clfs, imputation_methods)
np.save('results_exp1', scores_exp1)
mean_exp1 = np.mean(scores_exp1, axis=(2))
std_exp1 = np.std(scores_exp1, axis=(2))
for clf_id, clf_name in enumerate(clfs):
    for imputation_method_id, imputation_method in enumerate(imputation_methods):
        print("%s with %s imputation: %.3f (%.2f)" % (clf_name, imputation_method, mean_exp1[clf_id, imputation_method_id], std_exp1[clf_id, imputation_method_id]))


#testy dla Eksperymentu 1

scores_from_exp1 = np.load('results_exp1.npy')

alfa = .05
t_statistic_exp1 = np.zeros((len(clfs), len(imputation_methods)))
p_value_exp1 = np.zeros((len(clfs), len(imputation_methods)))
t_statistic_exp2 = np.zeros((len(clfs), len(imputation_methods)))
p_value_exp2 = np.zeros((len(clfs), len(imputation_methods)))
t_statistic_exp3 = np.zeros((len(clfs), len(imputation_methods)))
p_value_exp3 = np.zeros((len(clfs), len(imputation_methods)))

scores_exp1_clf1 = scores_from_exp1[0, :, :]
scores_exp1_clf2 = scores_from_exp1[1, :, :]
scores_exp1_clf3 = scores_from_exp1[2, :, :]

for i in range(len(clfs)):
    for j in range(len(clfs)):
        t_statistic_exp1[i, j], p_value_exp1[i, j] = ttest_rel(scores_exp1_clf1[i], scores_exp1_clf1[j])
        t_statistic_exp2[i, j], p_value_exp2[i, j] = ttest_rel(scores_exp1_clf2[i], scores_exp1_clf2[j])
        t_statistic_exp3[i, j], p_value_exp3[i, j] = ttest_rel(scores_exp1_clf3[i], scores_exp1_clf3[j])

#print("t-statistic:\n", t_statistic_exp1, "\n\np-value:\n", p_value_exp1)


headers = ["GNB median", "GNB mean", "GNB nearest"]
names_column = np.array([["GNB median"], ["GNB mean"], ["GNB nearest"]])
t_statistic_table = np.concatenate((names_column, t_statistic_exp1), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".4f")
p_value_table = np.concatenate((names_column, p_value_exp1), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".4f")
print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)


advantage = np.zeros((len(clfs), len(clfs)))
advantage[t_statistic_exp1 > 0] = 1
advantage_table = tabulate(np.concatenate(
(names_column, advantage), axis=1), headers)
print("Advantage:\n", advantage_table)

significance = np.zeros((len(clfs), len(clfs)))
significance[p_value_exp1 <= alfa] = 1
significance_table = tabulate(np.concatenate(
(names_column, significance), axis=1), headers)
print("Statistical significance (alpha = 0.05):\n", significance_table)

stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate(
(names_column, stat_better), axis=1), headers)
print("Statistically significantly better:\n", stat_better_table)

############################################################################################


print("\n\n")

headers1 = ["kNN median", "kNN mean", "kNN nearest"]
names_column1 = np.array([["kNN median"], ["kNN mean"], ["kNN nearest"]])
t_statistic_table1 = np.concatenate((names_column1, t_statistic_exp2), axis=1)
t_statistic_table1 = tabulate(t_statistic_table1, headers1, floatfmt=".4f")
p_value_table1 = np.concatenate((names_column1, p_value_exp2), axis=1)
p_value_table1 = tabulate(p_value_table1, headers1, floatfmt=".4f")
print("t-statistic:\n", t_statistic_table1, "\n\np-value:\n", p_value_table1)

advantage = np.zeros((len(clfs), len(clfs)))
advantage[t_statistic_exp2 > 0] = 1
advantage_table = tabulate(np.concatenate(
(names_column1, advantage), axis=1), headers1)
print("Advantage:\n", advantage_table)

significance = np.zeros((len(clfs), len(clfs)))
significance[p_value_exp2 <= alfa] = 1
significance_table = tabulate(np.concatenate(
(names_column1, significance), axis=1), headers1)
print("Statistical significance (alpha = 0.05):\n", significance_table)

stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate(
(names_column1, stat_better), axis=1), headers1)
print("Statistically significantly better:\n", stat_better_table)

############################################################################################

print("\n\n")
headers2 = ["CART median", "CART mean", "CART nearest"]
names_column2 = np.array([["CART median"], ["CART mean"], ["CART nearest"]])
t_statistic_table2 = np.concatenate((names_column2, t_statistic_exp3), axis=1)
t_statistic_table2 = tabulate(t_statistic_table2, headers2, floatfmt=".4f")
p_value_table2 = np.concatenate((names_column2, p_value_exp3), axis=1)
p_value_table2 = tabulate(p_value_table2, headers2, floatfmt=".4f")
print("t-statistic:\n", t_statistic_table2, "\n\np-value:\n", p_value_table2)

advantage = np.zeros((len(clfs), len(clfs)))
advantage[t_statistic_exp3 > 0] = 1
advantage_table = tabulate(np.concatenate(
(names_column2, advantage), axis=1), headers2)
print("Advantage:\n", advantage_table)

significance = np.zeros((len(clfs), len(clfs)))
significance[p_value_exp3 <= alfa] = 1
significance_table = tabulate(np.concatenate(
(names_column2, significance), axis=1), headers2)
print("Statistical significance (alpha = 0.05):\n", significance_table)

stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate(
(names_column2, stat_better), axis=1), headers2)
print("Statistically significantly better:\n", stat_better_table)

############################################################################################












