import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def process_input(df,scaler,selected_features):
    # model = joblib.load('credit_risk_logistic_regression_model.pkl')
    # scaler = joblib.load('scaler.pkl')
    # Ordinal encoding for categorical variables
    saving_mapping = {'little': 0, 'moderate': 1, 'quite rich': 2, 'rich': 3, 'unknown': -1}
    df['Saving accounts'] = df['Saving accounts'].map(saving_mapping)
    df['Checking account'] = df['Checking account'].map({'unknown': -1, 'little': 0, 'moderate': 1, 'rich': 2})

    # Housing_data.
    if df['Housing'].iloc[0] == 'own':
        df['Housing_own'] = 1
        df['Housing_rent'] = 0
    elif df['Housing'].iloc[0] == 'rent':
        df['Housing_own'] = 0
        df['Housing_rent'] = 1
    else:
        df['Housing_own'] = 0
        df['Housing_rent'] = 0

    df['Job'] = df['Job'].astype(int)
    # Scaling the features (using the existing scaler to avoid fitting again)
    df[['Age', 'Credit amount', 'Duration']] = scaler.transform(df[['Age', 'Credit amount', 'Duration']])

    # Deriving the new feature: Credit_Age_Ratio
    df['Credit_Age_Ratio'] = df['Credit amount'] / df['Age']
    df['Has_Saving_Account'] = df['Saving accounts'].apply(lambda x: 0 if x == -1 else 1)

    # Reordering the columns to match the training data
    df = df[selected_features]
    return df