import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
# from torchdata.datapipes.iter import IterableWrapper
import nltk
import pandas as pd
import numpy as np
from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams
nltk.download('words')
nltk.download('punkt')
import unicodedata
from nltk.corpus import words
correct_words = words.words()
device = torch.device("cpu")
#%%
df = pd.read_csv("/imdb/train.csv")
df['text_encoded'] = ''
df['split_word'] = df['text'].apply(nltk.word_tokenize)
#%%
def check_english_word(word):
    if word.isalpha():
        return True
    return False

def check_correct_word(word): # Correct words
    # https://www.geeksforgeeks.org/correcting-words-using-nltk-in-python/
    temp = [(jaccard_distance(set(ngrams(word, 2)), set(ngrams(w, 2))), w) for w in correct_words if w[0]==word[0]]
    return sorted(temp, key = lambda val:val[0])[0][1]

def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
#%%
df_copy = df.copy()
split_word = df_copy['split_word'].to_numpy() # Column to list
split_word_list = [j for i in split_word for j in i] # Split all words
split_word_list = [(unicodeToAscii(word)).lower() for word in split_word_list] # unicodeToAscii() and then lower()
split_word_list = [word for word in split_word_list if (len(word)>1) and (check_english_word(word))] # Extract only english words
split_word_list.remove("østbye")

# Now, We have a list include only english words
split_word_list_freq = (pd.Series(split_word_list).value_counts()).to_dict() # Create a dict that include frequency of each words
split_word_list_freq = {k:v for k, v in split_word_list_freq.items() if (int(v)>=20) and len(k)>1 and (check_english_word(k))} # Delete words with few frequency

i = 1
for k,v in split_word_list_freq.items():
    split_word_list_freq[k] = i
    i = i + 1
    
    
def df_text_to_index(df_col):
    ls = []
    for word in df_col:
        if split_word_list_freq.get(word) != None:
            get_index = split_word_list_freq.get(word)
            ls.append(get_index)
    return ls

df_copy['split_word_index'] = df_copy['split_word'].apply(df_text_to_index)
#%%

# df_copy['split_word_index'].str.len().agg(['max'])
t = torch.tensor([1,2,3,4,5,6])
t.cumsum(dim=0)
#%%
class imdb_dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df_ds = df.copy()

    def __getitem__(self, idx):
        x = self.df_ds['split_word_index'].iloc[idx]
        y = self.df_ds['sentiment'].iloc[idx]
        
        # x = torch.tensor([x])
        # y = torch.tensor([y])
        res = {'x':torch.tensor(x), 'y':y}
        return res

    def __len__(self):
        return len(self.df_ds)
#%%
from torch.nn.utils.rnn import pad_sequence

# def custom_collate(data): #(2)
#     inputs = [torch.tensor(d['x']) for d in data] #(3)
#     offsets = [0] + [len(entry) for entry in inputs]
#     offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
#     labels = [d['y'] for d in data]
#     inputs = pad_sequence(inputs, batch_first=True) #(4)
#     labels = torch.tensor(labels) #(5)
    
    
#     return inputs, labels, offsets

def custom_collate(data):
    inputs = [d['x'] for d in data]
    labels = [d['y'] for d in data]
    offsets = [0] + [len(entry) for entry in inputs]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    # inputs = pad_sequence(inputs, batch_first=True) #(4)
    labels = torch.tensor(labels) #(5)
    inputs = torch.cat(inputs)
    
    return inputs, labels, offsets
#%%
ds = imdb_dataset(df_copy)
dl = torch.utils.data.DataLoader(ds, batch_size=16, collate_fn=custom_collate,shuffle=False)
#%%
class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offset):
        embedded = self.embedding(text, offset)
        return self.fc(embedded)

net = TextSentiment(12799, 32, 2)

print("Start...")
tt = ''
labell = ''
def train_func(sub_train_):
    global tt; global labell
    train_loss = 0
    train_acc = 0
    data = torch.utils.data.DataLoader(sub_train_, batch_size=16, collate_fn=custom_collate, shuffle=False)
    for i, (x, y, offse) in enumerate(data):
        optimizer.zero_grad()
        offse = offse.to("cpu")
        x, y = x.to("cpu"), y.to("cpu")
        output = net(x.int(), offse)
        loss = criterion(output, y)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == y).sum().item()

    # Adjust the learning rate
    scheduler.step()

    return train_loss / len(sub_train_), train_acc / len(sub_train_)

def test(data_):
    loss = 0
    acc = 0
    data = torch.utils.data.DataLoader(data_, batch_size=16, collate_fn=custom_collate, shuffle=False)
    for i, (x, y, offse) in enumerate(data):
        offse = offse.to("cpu")
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            output = net(x.int(), offse)
            loss = criterion(output, y)
            loss += loss.item()
            acc += (output.argmax(1) == y).sum().item()

    return loss / len(data_), acc / len(data_)


import time
from torch.utils.data.dataset import random_split
N_EPOCHS = 50
min_valid_loss = float('inf')

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=4.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

train_len = int(len(ds) * 0.8)
sub_train_, sub_valid_ = random_split(ds, [train_len, len(ds) - train_len])

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    train_loss, train_acc = train_func(sub_train_)
    valid_loss, valid_acc = test(sub_valid_)

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')

    # if valid_loss < min_valid_loss:
    #     min_valid_loss = valid_loss
    #     torch.save(model, 'text_classification.pt')
#%%
sen = "Spirited was never going to be any good, but it would have been slightly better — and a change of pace — if Reynolds and Ferrell had switched roles."
lss = sen.split(" ")
ls2 = []
for word in lss:
    get_index = split_word_list_freq.get(word)
    if get_index != None:    
        ls2.append(get_index)
        
ls2 = torch.tensor(ls2)
with torch.no_grad():
    r = net(ls2, torch.tensor([0]))

if r.argmax(1) == 0:
    print("Like")
else:
    print("Dislike")
    
#%%
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import make_multilabel_classification
# This produces a feature matrix of token counts, similar to what
# CountVectorizer would produce on text.
X, y = make_multilabel_classification(random_state=0)
lda = LatentDirichletAllocation(n_components=6,
    random_state=0)
lda.fit(X)

# get topics for some given samples:
lda.transform(X[-2:])