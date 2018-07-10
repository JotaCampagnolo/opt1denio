'''
UNIVERSIDADE FEDERAL DA FRONTEIRA SUL - UFFS
Campus Chapecó - Ciência da Computação
GEX623 - Tópicos Especiais em Computação I (Machine Learning) - Matutino
Professor: Dr. Denio Duarte
Acadêmico: João Marcos Campagnolo
FINAL PROJECT: 3 Classifier Algorithms, using the Diabetic Dataset.
Article about the Dataset: https://www.hindawi.com/journals/bmri/2014/781670/
Note:   To clean the Dataset, i asked for help to Cleiton de Lima. He showed me the PANDAS library,
        and the ideia behind the discretizing of some features.
'''

# Imports:
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd

# DIABETIC_DATA.CSV Dataset features:
ftrs = [
    #('encounter_id', 'numeric'), REMOVED
    #('patient_nbr', 'numeric'), REMOVED
    ('race', 'nominal'),
    ('gender', 'nominal'),
    ('age', 'nominal'),
    #('weight', 'numeric'), REMOVED
    ('admission_type_id', 'nominal'),
    ('discharge_disposition_id', 'nominal'),
    ('admission_source_id', 'nominal'),
    ('time_in_hospital', 'numeric'),
    #('payer_code', 'nominal'), REMOVED
    ('medical_specialty', 'nominal'),
    ('num_lab_procedures', 'numeric'),
    ('num_procedures', 'numeric'),
    ('num_medications', 'numeric'),
    ('number_outpatient', 'numeric'),
    ('number_emergency', 'numeric'),
    ('number_inpatient', 'numeric'),
    ('diag_1', 'nominal'),
    ('diag_2', 'nominal'),
    ('diag_3', 'nominal'),
    ('number_diagnoses', ''),
    ('max_glu_serum', 'nominal'),
    ('A1Cresult', 'nominal'),
    ('metformin', 'nominal'),
    ('repaglinide', 'nominal'),
    ('nateglinide', 'nominal'),
    ('chlorpropamide', 'nominal'),
    ('glimepiride', 'nominal'),
    ('acetohexamide', 'nominal'),
    ('glipizide', 'nominal'),
    ('glyburide', 'nominal'),
    ('tolbutamide', 'nominal'),
    ('pioglitazone', 'nominal'),
    ('rosiglitazone', 'nominal'),
    ('acarbose', 'nominal'),
    ('miglitol', 'nominal'),
    ('troglitazone', 'nominal'),
    ('tolazamide', 'nominal'),
    ('examide', 'nominal'),
    ('citoglipton', 'nominal'),
    ('insulin', 'nominal'),
    ('glyburide-metformin', 'nominal'),
    ('glipizide-metformin', 'nominal'),
    ('glimepiride-pioglitazone', 'nominal'),
    ('metformin-rosiglitazone', 'nominal'),
    ('metformin-pioglitazone', 'nominal'),
    ('change', 'nominal'),
    ('diabetesMed', 'nominal'),
    ('readmitted', 'nominal')
]

# Preprocessing function that cleans the Dataset:
def cleandataset(dirtdataset):
    # Replaces the '?' values to 'NaN' from NumPy:
    dirtdataset = dirtdataset.replace('?', np.NaN)
    # Includes the 'missing' value to the null ones in MEDICAL_SPECALTY:
    dirtdataset['medical_specialty'].fillna('missing', inplace=True)
    # Deletes the duplicated patients: "in particular, we considered only the first encounter
    # for each patient as the primary admission and determined whether or not they were readmitted
    # within 30 days."
    # This information is in the article about the Dataset: https://www.hindawi.com/journals/bmri/2014/781670/
    dirtdataset = dirtdataset.drop_duplicates('patient_nbr', keep='first')
    # Removes the features WEIGHT (lot of missing values) and ENCOUNTER_ID, PATIENT_NBR and PAYER_CODE (relevance):
    dirtdataset.drop(['encounter_id', 'patient_nbr', 'weight', 'payer_code'], inplace=True, axis=1)
    # For null values from RACE, DIAG_1, DIAG_2 and DIAG_3, the most recorrent value is assigned:
    dirtdataset['race'].fillna(dirtdataset['race'].mode()[0], inplace=True)
    dirtdataset['diag_1'].fillna(dirtdataset['diag_1'].mode()[0], inplace=True)
    dirtdataset['diag_2'].fillna(dirtdataset['diag_2'].mode()[0], inplace=True)
    dirtdataset['diag_3'].fillna(dirtdataset['diag_3'].mode()[0], inplace=True)
    # Convert Numinal features to Numerics:
    numerics = [n[0] for n in ftrs if n[1] == 'numeric']
    nominals = [n[0] for n in ftrs if n[1] == 'nominal']
    df_num = dirtdataset[numerics] # Separate the Numeric ones.
    df_nom = dirtdataset[nominals] # Separate the Nominal ones.
    df_nom = df_nom.apply(LabelEncoder().fit_transform) # Convert the Nominal ones.
    dirtdataset = pd.concat([df_num, df_nom], axis=1) # Concat all features again.
    # Returns the clean Dataset:
    return dirtdataset

# Program Start PRINT:
print("[0] : The PROGRAM has started!\n")

# Openning INPUT FILE:
inputFILE = pd.read_csv('diabetic_data.csv', sep=',')
print("[1] : Input FILE was successfully oppened!\n")
print(inputFILE.shape)

# Cleaning the Dataset:
inputDATA = cleandataset(inputFILE)
print("[2] : Input DATA was successfully cleaned!\n")

# Instanciation of DATA matrices:
X = inputDATA.iloc[:, :-1]
y = inputDATA.iloc[:, -1]
print("[3] : Input DATA matrices was successfully created!")
print("      *", X.shape[0], "samples.")
print("      *", X.shape[1], "features.\n")

# Identifying the 10 best features to create the model:
X = SelectKBest(k=10).fit_transform(X, y)
print("[4] : 10 best features was successfully indentified!\n")

# Separate the TRAIN and TEST samples:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)
print("[5] : The TRAIN and TEST samples was successfully separated!\n")

# Selecting the Classifiers Algorithms:
# Algorithms choosed by https://www.kaggle.com/jeffd23/10-classifier-showdown-in-scikit-learn analysis.
DT = DecisionTreeClassifier()
KNN = KNeighborsClassifier()
LDA = LinearDiscriminantAnalysis()
CF_LIST = [DT, KNN, LDA]
print("[6] : The selected algorithms was:")
print("      * Decision Tree Classifier;")
print("      * K Neighbors Classidier;")
print("      * Linear Discriminant Analysis.\n")

# Defining the hyper parameters for each algorithm:
DT_HP = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_features': ['auto', 'log2'],
    'random_state' : [2015412]
} # http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
KNN_HP = {
    'n_neighbors': [10, 12],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'kd_tree'],
    'leaf_size': [25, 30, 35]
} # http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
LDA_HP = {
    'solver': ['lsqr', 'eigen'],
    'shrinkage': ['auto', 0.5],
    'n_components': [1, 3]
} # http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
HP_LIST = [DT_HP, KNN_HP, LDA_HP]

# Running the classifiers:
for CF, HP in zip(CF_LIST, HP_LIST):
    grid = GridSearchCV(CF, HP, cv=5) # 'cv=5' defined by professor.
    # Running the algorithm:
    print('RUNNING: {} algorithm:'.format(CF.__class__.__name__))
    model = grid.fit(X_train, y_train)
    # Printing the best parameters found:
    print('BEST HYPER PARAMETERS:')
    print(model.best_estimator_)
    # Running the model to TEST Dataset:
    y_hat = model.predict(X_test)
    print('FOUND RESULTS:')
    print(classification_report(y_test, y_hat))
