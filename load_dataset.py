## loading libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import csv
import numpy as np

with open('./dataset/common-surnames-by-country.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    countries = []
    names = []
    for i, row in enumerate(spamreader):
        rowlist=row[0].split(',')
        # print(rowlist)
        if len(rowlist)>6:
            country = rowlist[0]
            name = rowlist[5]
            if name != '':
                countries.append(country)
                names.append(name)
                
vocab_size = len(set(''.join(names)))
cty_size = len(set(countries))

ctytoi = {c:i for i, c in enumerate(sorted(list(set(countries))))}
itocty = {i:c for i, c in enumerate(sorted(list(set(countries))))}
chars = ['.']+sorted(list(set(''.join(names))))
vocab_size += 1
stoi = {s: i for i, s in enumerate(chars)}
itos = {i: s for s, i in stoi.items()}

L = list(range(len(names)))
np.random.shuffle(L)
countries = [countries[l] for l in L]
names = [names[l] for l in L]
block_size = 3

def load_dataset(names, countries, block_size = 3):
    X, C, Y = [], [], []
    for w, c in zip(names, countries):
        context = [0]*block_size
        ic = ctytoi[c]
        for ch in w+'.':
            ix = stoi[ch]
            X.append(context)
            C.append(ic)
            Y.append(ix)
            context = context[1:]+[ix]
    X = torch.tensor(X)
    C = torch.tensor(C)
    Y = torch.tensor(Y)
    return X, C, Y

n1 = int(0.9*len(names))
Xtr, Ctr, Ytr = load_dataset(names[:n1], countries[:n1], block_size)
Xde, Cde, Yde = load_dataset(names[n1:], countries[n1:], block_size)

