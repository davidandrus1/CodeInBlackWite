import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

def normalize_position(df, position, features, save_path="saved_models/"):
    
    X = df[features]
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    df_scaled = pd.DataFrame(X_scaled, columns=features, index=df.index)
    df_scaled.insert(0, "playerId", df["playerId"].values)

    # Salvezi scaler-ul pentru jucători noi
    os.makedirs(save_path, exist_ok=True)
    with open(f"{save_path}scaler_{position}.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return df_scaled