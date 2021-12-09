from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np

def build_dictionary(word_list):
    # from https://github.com/HKUST-KnowComp/JWE/blob/master/src/word_sim.py
    dictionary = dict()
    cnt = 0
    for w in word_list:
        dictionary[w] = cnt
        cnt += 1
    return dictionary

def read_vectors(vec_file):
    # from https://github.com/HKUST-KnowComp/JWE/blob/master/src/word_sim.py
    # input:  the file of word2vectors
    # output: word dictionary, embedding matrix -- np ndarray
    f = open(vec_file,'r')
    cnt = 0
    word_list = []
    embeddings = []
    word_size = 0
    embed_dim = 0
    for line in f:
        data = line.split()
        if cnt == 0:
            word_size = data[0]
            embed_dim = data[1]
        else:
            word_list.append(data[0])
            tmpVec = [float(x) for x in data[1:]]
            embeddings.append(tmpVec)
        cnt = cnt + 1
    f.close()
    embeddings = np.array(embeddings)
    for i in range(int(word_size)):
        embeddings[i] = embeddings[i] / np.linalg.norm(embeddings[i])
    dict_word = build_dictionary(word_list)
    return word_size, embed_dim, dict_word, embeddings

def parse_char2comp(char2comp_filepath):
    # create Chinese character to index mapping dictionary
    # and list of all subcomponent decompositions
    with open(char2comp_filepath) as f:
        lines = f.readlines()

    char_list = []
    subcomponent_list = []
    for idx, line in enumerate(lines):
        line = line.rstrip('\n')
        x = line.split(' ')
        
        _char = x[0]
        _char_subcomponents = x[1::]

        char_list.append(_char)
        subcomponent_list.append(_char_subcomponents)

    # append UNK 
    char_list.append("UNK")
    subcomponent_list.append(["UNK"])
    
    char2id = {v:i for i,v in enumerate(char_list)}
    return char2id, subcomponent_list

# removing punctuation
def char_tokenizer(text):
    return text.translate(str.maketrans("", "", '!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~“”‘’…！，。、~'))

# mapping sentence to char indices
def text2charidx(text, char2id):
    tokenized_text = [c for c in char_tokenizer(text)]
    return [char2id[c] if c in char2id else char2id["UNK"] for c in tokenized_text]

# finding the corresponding subcomponents, mapping to idx from the trained data
def text2subcomponent(text, subcomponent_list, comp2id, char2id, unk_idx):
    char_idx_list = text2charidx(text, char2id)
    subcomp = [subcomponent_list[idx] for idx in char_idx_list]
    output = []
    for x in subcomp:
      output.extend([comp2id[s] if s in comp2id else unk_idx for s in x])
    return output

# remove unks from list of lists of subcomponent ids:
def remove_unks(subcomponent_ids, unk_index):
    output = []
    for obs in subcomponent_ids:
        out = [i for i in obs if i!= unk_index]
        output.append(out)
    return output

# truncate or pad each list in the list subcomponent_ids to pad_length:
def pad_entries(subcomponent_ids, pad_length, pad_index, unk_index):
    padded = []
    for obs in subcomponent_ids:
        obs = [i for i in obs if i!= unk_index]
        out = obs[:pad_length] #truncate
        if len(out) < pad_length:
            out.extend([pad_index]*(pad_length - len(out)))
        padded.append(out)
    return padded

# parse the file with trained mapping from components to vectors
def parse_comp2vec(comp2vec_filepath):
    comp_vocab_size, comp_embedding_size, comp2id, comp_embeddings = read_vectors(comp2vec_filepath)
    pad_idx = int(comp_vocab_size)
    unk_idx = int(comp_vocab_size) + 1
    comp2id["UNK"] = unk_idx
    comp_embeddings = np.vstack((comp_embeddings, np.zeros((1, np.shape(comp_embeddings)[1]))))
    comp_embedding_size = int(comp_embedding_size)
    print(f"Component embedding shape: {comp_embeddings.shape}")
    print(f"UNK idx reserved for id: {unk_idx}")
    print(f"PAD idx reserved for id: {pad_idx}")
    return comp_vocab_size, comp_embedding_size, comp2id, comp_embeddings, pad_idx, unk_idx

# preprocess list of text by convrting to list of lists of component_ids
def decompose(X_list, subcomponent_list, comp2id, char2id, unk_idx, pad_idx, pad_length = None):
    component_ids = [text2subcomponent(i, subcomponent_list, comp2id, char2id, unk_idx) for i in X_list]
    component_ids = remove_unks(component_ids, unk_idx)
    max_decomposition_length = np.max(np.array([len(i) for i in component_ids]))
    print(f"Maximum decomposition length: {max_decomposition_length}")
    if pad_length:
        component_ids = pad_entries(component_ids, pad_length, pad_idx, unk_idx)
    else:
        component_ids = pad_entries(component_ids, max_decomposition_length, pad_idx, unk_idx)
    return component_ids, max_decomposition_length

# Define subcharacter info dataset class
# based on: https://huggingface.co/transformers/custom_datasets.html
class ComponentDataset(Dataset):
    def __init__(self, encodings, labels, subcomponent_ids):
        self.encodings = encodings
        self.labels = labels
        self.subcomponent_ids = subcomponent_ids

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['subcomponent_ids'] = torch.tensor(self.subcomponent_ids[idx])
        return item

    def __len__(self):
        return len(self.labels)