from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline


def train_model(df, target='Survived'):
    # Extracting features from train_data
    X = df.drop(columns =[target])
    y = df[target]
    # Splitting the train_data to train set and test set
    svc = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel='rbf', C=1, probability=True))
    score1=cross_val_score(svc,X,y).mean()
    rfc=RandomForestClassifier(n_estimators = 70)
    score2=cross_val_score(rfc,X,y).mean()
    # Create an esemble for improved accuracy
    vc=VotingClassifier([('clf1',svc),('clf2',rfc)],voting='soft')
    score3=cross_val_score(vc,X,y).mean()
    vc.fit(X,y)
    return vc