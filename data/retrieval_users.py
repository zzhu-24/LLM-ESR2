# %%
import os
import pickle
import numpy as np
np.random.seed(42)
import pandas as pd
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
# import matplotlib.pyplot as plt
import argparse

# %%
parser = argparse.ArgumentParser(description='Generate user embeddings')
parser.add_argument('--enable_id', action='store_true', 
                       help='Enable ID in user embeddings')
args = parser.parse_args()

# %%
dataset = "beauty2014"
sim_metric = "cos"
topk = 100

# %% [markdown]
# ### Get the topk similar user

# %%
# load the llm user embedding
if args.enable_id:
    user_emb = pickle.load(open(os.path.join(dataset+"/handled/", "usr_emb_collab_np.pkl"), "rb"))
else:
    user_emb = pickle.load(open(os.path.join(dataset+"/handled/", "usr_emb_np.pkl"), "rb"))

# %%
# calculate the similarity score between users based on llm user embedding
if sim_metric == "sin":
    score_matrix = np.dot(user_emb, user_emb.T)
elif sim_metric == "cos":
    score_matrix = cosine_similarity(user_emb, user_emb)

# %%
# plt.hist(score_matrix[0], bins=10)
# plt.show()

# %%
rank_matrix = np.argsort(-score_matrix, axis=-1)    # user id starts from 0

# %%
final_rank_matrix = rank_matrix[:, 1:]
final_rank_matrix = final_rank_matrix[:, :topk]

# %% [markdown]
# ### Get the sequence length of each user

# %%
User = defaultdict(list)
seq_len = []
usernum, itemnum = 0, 0
f = open('./%s/handled/%s.txt' % (dataset, "inter"), 'r')
for line in f:  # use a dict to save all seqeuces of each user
    u, i = line.rstrip().split(' ')
    u = int(u)
    i = int(i)
    usernum = max(u, usernum)
    itemnum = max(i, itemnum)
    User[u].append(i)

for user, seq in User.items():
    seq_len.append(len(seq))

# %%
sim_user_len = []
for sim_user_list in final_rank_matrix:
    avg_len = 0
    for sim_user in sim_user_list:
        avg_len += seq_len[sim_user] / topk
    sim_user_len.append(avg_len)

# %%
print(np.mean(sim_user_len), np.mean(seq_len))

# %% [markdown]
# ### Select the similar user

# %%
sim_users = []
for sim_user_list in final_rank_matrix:
    sim_users.append(np.random.choice(sim_user_list, 1)[0])

# %%
print(final_rank_matrix.shape)

# %%
## Save llm embedding based similar users
if args.enable_id:
    pickle.dump(final_rank_matrix, open(os.path.join(dataset+"/handled/", "sim_user_collab_100.pkl"), "wb"))
else:
    pickle.dump(final_rank_matrix, open(os.path.join(dataset+"/handled/", "sim_user_100.pkl"), "wb"))
