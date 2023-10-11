import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from alibi_detect.cd import MMDDriftOnline
from alibi_detect.saving import save_detector, load_detector
import joblib
import json

df = pd.read_csv('creditcard_2023.csv', index_col='id')

test_size = 0.1

num_test_samples = int(len(df) * test_size)
test_indices = df.sample(n=num_test_samples, random_state=42).index

train_df = df.drop(index=test_indices)
test_df = df.loc[test_indices]

clf = RandomForestClassifier(n_jobs=2)
clf.fit(train_df.drop("Class", axis=1).values, train_df["Class"].values)

joblib.dump(clf, 'base_model/model.joblib')

train_df.to_csv('base_model/training_dataset.csv', index=False)
test_df.to_csv('base_model/testing_dataset.csv', index=False)

#drift_detector = MMDDriftOnline(train_df.values[:10000,:], 25, 10, n_bootstraps=250, backend="pytorch") #TODO fix memory consumption problems
#save_detector(drift_detector, 'base_model/online-MMD')

# Create metadata dictionary
metadata = {
    'model': str(type(clf)),
    'n_jobs': str(clf.n_jobs),
    'used_columns': [str(column) for column in train_df.drop("Class", axis=1).columns],
    'classes': [str(class_) for class_ in clf.classes_],
    'test_size': str(test_size),
#    'drift_detectors': [
#        {
#            'model': str(type(drift_detector)),
#            'ert': '10',
#            'window_size': '100',
#            'n_bootstraps': '10',
#            'backend': 'pytorch'
#        }
#    ]
}

with open('base_model/metadata.json', 'w') as json_file:
    json.dump(metadata, json_file)

