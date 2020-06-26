import pickle

with open('test_statistics.pkl','rb') as f:
    test_statistics = pickle.load(f)

with open('train_statistics.pkl', 'rb') as f:
    train_statistics = pickle.load(f)

