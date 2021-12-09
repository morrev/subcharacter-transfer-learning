from sklearn.model_selection import train_test_split
import pickle

def load_data(path):
    with open(path, "rb") as bf:
        data = pickle.load(bf)
    return data

def load_livedoor():
    train_dataset = load_data('data/GDCE-SSA/data/pickle/train_livedoor.pkl')['data']
    test_dataset = load_data('data/GDCE-SSA/data/pickle/test_livedoor.pkl')['data']
    X_train, y_train = list(zip(*train_dataset))
    X_test, y_test = list(zip(*test_dataset))
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify = y_train,
                                                  test_size = 0.2, random_state = 42)
    return X_train, X_val, X_test, y_train, y_val, y_test
