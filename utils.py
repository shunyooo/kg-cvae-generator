import json
import pickle


def load_pickle_class(path):
    with open(path, 'rb') as file:
        dv = pickle.load(file)
    return dv


def save_pickle_class(path, obj):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)
    print('save this class in', path)


def load_config(path):
    with open(path, 'r', encoding='utf-8') as file:
        output = json.load(file)
    return output


def save_config(path, target):
    with open(path, 'w') as fp:
        json.dump(target, fp)
