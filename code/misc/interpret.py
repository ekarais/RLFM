import numpy as np

def policy(S_t, Q):
    return np.argmax(Q[S_t])*10


Q = np.load('Q_values.npy')


for i in range(0, 11):
    print(f"Last offer = {i*10}  ==> Next offer = {policy(i, Q)}")