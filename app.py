
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder


with open('Naive_Baye.pkl', 'rb') as f:
    BNB_model = pickle.load(f)

import pandas as pd

data = [
    [0,"tcp","supdup","S0",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,205,17,1,1,0,0,0.08,0.06,0,255,17,0.07,0.07,0,0,1,1,0,0]
]

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count',
    'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

df = pd.DataFrame(data, columns=columns)


features = ['protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'count', 'dst_host_same_srv_rate','same_srv_rate','diff_srv_rate','dst_host_same_srv_rate']



import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(train, test):
    # Label encoding
    le = LabelEncoder()
    for col in train.columns:
        if train[col].dtype == 'object':
            le.fit(train[col])
            train[col] = le.transform(train[col])
            test[col] = le.transform(test[col])

    # Feature selection
    X_train = train.drop(['class'], axis=1)
    Y_train = train['class']
    rfc = RandomForestClassifier()
    rfe = RFE(rfc, n_features_to_select=10)
    rfe = rfe.fit(X_train, Y_train)

    feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), X_train.columns)]
    selected_features = [v for i, v in feature_map if i==True]

    X_train = X_train[selected_features]
    X_test = test[selected_features]

    # Scaling
    scale = StandardScaler()
    X_train = scale.fit_transform(X_train)
    X_test = scale.transform(X_test)

    return X_train, X_test, Y_train

X_scaled,y,s = preprocess_data(df, df)
print(X_scaled)
X_scaled = X_scaled.reshape(1, -1)
print(X_scaled)

s=[]
s.append(X_scaled)
print(BNB_model.predict(X_scaled))
