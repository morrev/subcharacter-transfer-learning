from sklearn.model_selection import train_test_split
import pandas as pd
import pickle, json

def load_data(path):
    with open(path, "rb") as bf:
        data = pickle.load(bf)
    return data

def load_livedoor(colab = False):
    if colab:
        prepath = "/content/drive"
    else:
        prepath = "."
    train_dataset = load_data(f'{prepath}/data/GDCE-SSA/data/pickle/train_livedoor.pkl')['data']
    test_dataset = load_data(f'{prepath}/data/GDCE-SSA/data/pickle/test_livedoor.pkl')['data']
    X_train, y_train = list(zip(*train_dataset))
    X_test, y_test = list(zip(*test_dataset))
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify = y_train,
                                                  test_size = 0.2, random_state = 42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_wikipedia(colab = False):
    if colab:
        prepath = "/content/drive"
    else:
        prepath = "."
    data_path = f'{prepath}/data/Wikipedia_title_dataset/ja_raw.txt'
    with open(data_path) as f:
        data = json.load(f)
    
    data_df = pd.DataFrame(data).T 
    train_idx, test_idx = train_test_split(data_df.index, test_size=0.2, stratify=data_df.category, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, stratify=data_df.loc[train_idx].category, random_state=42)

    X_train, y_train = data_df.loc[train_idx][['title', 'category']].values.T
    X_val, y_val = data_df.loc[val_idx][['title', 'category']].values.T
    X_test, y_test = data_df.loc[test_idx][['title', 'category']].values.T

    X_train = list(X_train)
    X_val = list(X_val)
    X_test = list(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def write_pickle(path, d):
    try:
      with open(path,'wb') as f:
          return pickle.dump(d, f, protocol = pickle.HIGHEST_PROTOCOL)
    except:
        print(f'Write pickle error on {f}')
