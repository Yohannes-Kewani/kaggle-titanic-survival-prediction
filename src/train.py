from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def train_model(df, target='Survived'):
    # Extracting features from train_data
    X = df.drop(columns =[target])
    y = df[target]
    # Splitting the train_data to train set and test set
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    # create a linear Regresion 
    model= LinearRegression()
    # Train the model
    model.fit(X_train,y_train)
    return model, X_test, y_test