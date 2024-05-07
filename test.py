import itertools
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

import streamlit as st
st.set_page_config(layout="wide")

st.title("Anomaly detection in a network")


st.file_uploader("Upload a csv containing the apporpriate information",type=["csv"],key="file")







def le(df):
    for col in df.columns:
        if df[col].dtype == 'object':
                label_encoder = LabelEncoder()
                df[col] = label_encoder.fit_transform(df[col])

def preprocess_data(train, test):
    le(train)
    le(test)
    print(train.head())

    train.drop(['num_outbound_cmds'], axis=1, inplace=True)
    test.drop(['num_outbound_cmds'], axis=1, inplace=True)

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

    return X_train, X_test, Y_train, le

def process_data(data):
    le(data)
    
    print(data.head())
    selected_features=['protocol_type', 'service',  'flag',  'src_bytes',  'dst_bytes',  'count',  'same_srv_rate',  'diff_srv_rate',  'dst_host_srv_count',  'dst_host_same_srv_rate']
    X_train = data[selected_features]
    scale = StandardScaler()
    X_train = scale.fit_transform(X_train)


    return X_train



train=pd.read_csv('Train_data.csv')
test=pd.read_csv('Test_data.csv')






with open('Naive_Baye.pkl', 'rb') as f:
    BNB_model = pickle.load(f)



if st.session_state.file:
     st.info("1= anomaly, 0= normal")
     col1, col2, = st.columns(2)

     s= pd.read_csv(st.session_state.file)
     X_train = process_data(s)
     st.balloons()
     with col1:
        st.write(s)
     with col2:
        st.write(BNB_model.predict(X_train))