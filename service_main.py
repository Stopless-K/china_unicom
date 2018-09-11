import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from data_provider import load
from config import Config
config = Config()
def train(x, y):

    forest_clf = RandomForestClassifier(n_estimators=20, n_jobs=-1, random_state=0)
    forest_clf.fit(x, y)
    return forest_clf

def evaluate(clf, x, y):

    predict = clf.predict(x)

    acc = np.mean(predict == y)

    print(acc)

def predict(x, clf):
    pred = clf.predict(x)
    idx_2_service = pd.read_pickle(config.out_path+ config.idx_2_service)

    pred = [idx_2_service[i] for i in pred]

    return pred

def write_csv(pred, user_id):
    df = pd.DataFrame({'user_id': user_id,  'predict': pred})
    df.to_csv(config.out_path+ 'pred_result.csv', index=None)
def run():

    x, y = load('train')
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0, random_state=0)
    clf = train(x_train, y_train)

    # evaluate(clf, x_val, y_val)

    x_test, user_id = load('test')
    pred = predict(x_test, clf)

    write_csv(pred, user_id)




if __name__ == '__main__':
    run()
