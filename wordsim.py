from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import argparse
import torch
import csv
from models import *
from wordsim_utils import *
from transformers import BertConfig, AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from decomposition_utils import *
from glyph_utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--model",
    help = "model: \"baseline\", \"glyph\", \"radical\", or \"subcomponent\"",
    default = "baseline")
parser.add_argument("--frozen",
    help = "whether to freeze subcomponent/radical embeddings",
    default = "False")
parser.add_argument("--pooled",
    help = "pooled or unpooled model, default pooled",
    default = "True")
parser.add_argument("--dataset",
    help = "finetuning dataset, either \"livedoor\" or \"wiki\"",
    default = "livedoor")
parser.add_argument("--num_trials",
    help = "number of trials to run for",
    default = "10")
parser.add_argument("--train_proportion",
    help = "proportion of dataset to be used for training, decimal format, 0 to 1",
    default = "0.8")
args = parser.parse_args()

# constants
BATCH_SIZE = 1

# load JWSAN data
wstxtfile = 'data/JWSAN/jwsan-1400.csv'
w1s = []
w2s = []
scores = []
with open(wstxtfile, "r") as o:
    reader = csv.reader(o)
    for row in reader:
      w1s.append(row[1])
      w2s.append(row[2])
      scores.append(row[4])
w1s = w1s[1:]
w2s = w2s[1:]
scores = scores[1:]
tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-char-v2")
encodings_w1 = tokenizer(w1s, truncation=True, padding=True, max_length = 512)
encodings_w2 = tokenizer(w2s, truncation=True, padding=True, max_length = 512)

# create dataset
class WordSimDataset(Dataset):
    def __init__(self, 
                 encodings,
                 additional_embeddings = None, 
                 seq_lengths = None):
        self.encodings = encodings
        self.seq_lengths = seq_lengths
        self.additional_embeddings = additional_embeddings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['additional_embeddings'] = torch.tensor(self.additional_embeddings[idx]).float() if self.additional_embeddings else None
        item['seq_lengths'] = torch.tensor(self.seq_lengths[idx]) if self.seq_lengths else None
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

# load model
h1, h2 = None, None
if args.model == "glyph":
    if args.pooled == "False":
        seq_lengths_w1, glyph_embs_w1 = text2glyph(w1s)
        seq_lengths_w2, glyph_embs_w2 = text2glyph(w2s)
        dataset_w1 = WordSimDataset(encodings_w1, glyph_embs_w1, seq_lengths_w1)
        dataset_w2 = WordSimDataset(encodings_w2, glyph_embs_w2, seq_lengths_w2)
        with open(f"models/unpooled_jbert_cbert_glyph_{args.dataset}.pk", "rb") as f:
            model = pickle.load(f).to(device)
    else:
        glyph_embeddings_w1 = get_glyph_embeddings(w1s, True)
        glyph_embeddings_w2 = get_glyph_embeddings(w2s, True)
        dataset_w1 = WordSimDataset(encodings_w1, glyph_embeddings_w1)
        dataset_w2 = WordSimDataset(encodings_w2, glyph_embeddings_w2)
        GLYPH_EMBEDDING_SIZE = 1728
        model = CustomPooledModel.from_pretrained("cl-tohoku/bert-base-japanese-char-v2", num_labels = 9).to(device)
        model.load_state_dict(torch.load(f"models/bert-base-japanese-{args.dataset}-JWE-glyph/pytorch_model.bin"))
        h1 = model.glyph_embeddings.register_forward_hook(getActivation('glyph_embeddings'))
        h2 = model.bert.pooler.register_forward_hook(getActivation('pooler'))
elif args.model == "radical" or args.model == "subcomponent":
    char2comp_fpath = "data/JWE/subcharacter/char2" + "comp.txt" if args.model == "subcomponent" else "radical.txt"
    vec_dir = f'data/JWE-pretrained/{args.model}_comp_vec'
    comp_size, SUBCOMPONENT_EMBEDDING_SIZE, comp2id, SUBCOMPONENT_EMBEDDINGS = read_vectors(vec_dir)
    UNK_IDX = int(comp_size) # Set UNK to be vocab size + 1
    char2id, subcomponent_list = parse_char2comp(char2comp_fpath)
    comp2id['UNK'] = UNK_IDX
    SUBCOMPONENT_EMBEDDINGS = np.vstack((SUBCOMPONENT_EMBEDDINGS, np.zeros((1, np.shape(SUBCOMPONENT_EMBEDDINGS)[1]))))
    SUBCOMPONENT_EMBEDDING_SIZE = int(SUBCOMPONENT_EMBEDDING_SIZE)
    subcomponent_ids_w1 = [text2subcomponent(i, subcomponent_list, comp2id, char2id, unk_idx = UNK_IDX, tokenizer=tokenizer) for i in w1s]
    subcomponent_ids_w2 = [text2subcomponent(i, subcomponent_list, comp2id, char2id, unk_idx = UNK_IDX, tokenizer=tokenizer) for i in w2s]
    seq_lengths_w1 = [len(ids) for ids in subcomponent_ids_w1]
    seq_lengths_w2 = [len(ids) for ids in subcomponent_ids_w2]
    SEQ_LEN_W1 = len(encodings_w1['input_ids'][0])
    SEQ_LEN_W2 = len(encodings_w2['input_ids'][0])
    subcomponent_embs_w1 = subcomponent2emb(subcomponent_ids_w1, padding = True, seq_length = SEQ_LEN_W1)
    subcomponent_embs_w2 = subcomponent2emb(subcomponent_ids_w2, padding = True, seq_length = SEQ_LEN_W2)
    if args.pooled == "False":
        GLYPH_EMBEDDING_SIZE = 1024
        dataset_w1 = WordSimDataset(encodings_w1, subcomponent_embs_w1, seq_lengths_w1)
        dataset_w2 = WordSimDataset(encodings_w2,subcomponent_embs_w2, seq_lengths_w2)
        with open(f"models/unpooled_jbert_cbert_{args.model}_{args.dataset}.pk", "rb") as f:
            model = pickle.load(f).to(device)
    else:
        dataset_w1 = WordSimDataset(encodings_w1, subcomponent_ids_w1)
        dataset_w2 = WordSimDataset(encodings_w2, subcomponent_ids_w2)
        model = CustomPooledModel.from_pretrained("cl-tohoku/bert-base-japanese-char-v2", num_labels = 9).to(device)
        if model.frozen == "True":
            config = BertConfig.from_json_file(f"models/bert-base-japanese-{args.dataset}-JWE-{args.model}-frozen/config.json")
            model.load_state_dict(torch.load(f"models/bert-base-japanese-{args.dataset}-JWE-{args.model}-frozen/pytorch_model.bin"))
        else:
            config = BertConfig.from_json_file(f"models/bert-base-japanese-{args.dataset}-JWE-{args.model}/config.json")
            model.load_state_dict(torch.load(f"models/bert-base-japanese-{args.dataset}-JWE-{args.model}/pytorch_model.bin"))  
        h1 = model.subcomponent_embedding.register_forward_hook(getActivation('subcomponent_embedding'))
        h2 = model.bert.pooler.register_forward_hook(getActivation('pooler'))
else:
    dataset_w1 = WordSimDataset(encodings_w1)
    dataset_w2 = WordSimDataset(encodings_w2)
    config = BertConfig.from_json_file(f"models/bert-base-japanese-{args.dataset}-titles/config.json")
    model = AutoModelForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese-char-v2", num_labels = 9).to(device)
    model.load_state_dict(torch.load(f"models/bert-base-japanese-{args.dataset}-titles/pytorch_model.bin"))
    h2 = model.bert.pooler.register_forward_hook(getActivation('pooler'))
loader_1 = DataLoader(dataset_w1, batch_size = BATCH_SIZE, shuffle=True)
loader_2 = DataLoader(dataset_w2, batch_size = BATCH_SIZE, shuffle=True)
model.eval()

# get embeddings
all_logits_w1 = []
all_logits_w2 = []
with torch.no_grad():
    for batch in loader_1:
        interm = intermediate_output(batch, model, pooled = bool(args.pooled))
        all_logits_w1.append(interm.cpu())
    for batch in loader_2:
        interm = intermediate_output(batch, model, pooled = bool(args.pooled))
        all_logits_w2.append(interm.cpu())
if h1:
    h1.remove()
if h2:
    h2.remove()

# run linear regression
mean = 0
ntrials = int(args.num_trials)
for j in range(ntrials):
  training_in = []
  training_ids = random.sample([i for i in range(len(w1s))], int(len(w1s) * float(args.train_proportion)))
  testing_ids = [i for i in range(len(w1s)) if i not in training_ids]
  for i in training_ids:
    cos = cos_sim(all_logits_w1[i].squeeze(0).squeeze(0), all_logits_w2[i].squeeze(0).squeeze(0))
    euc = euclidean(all_logits_w1[i].squeeze(0).squeeze(0), all_logits_w2[i].squeeze(0).squeeze(0))
    manh = manhattan(all_logits_w1[i].squeeze(0).squeeze(0), all_logits_w2[i].squeeze(0).squeeze(0))
    training_in.append([cos, euc, manh])
  testing_in = []
  for i in testing_ids:
    cos = cos_sim(all_logits_w1[i].squeeze(0).squeeze(0), all_logits_w2[i].squeeze(0).squeeze(0))
    euc = euclidean(all_logits_w1[i].squeeze(0).squeeze(0), all_logits_w2[i].squeeze(0).squeeze(0))
    manh = manhattan(all_logits_w1[i].squeeze(0).squeeze(0), all_logits_w2[i].squeeze(0).squeeze(0))
    testing_in.append([cos, euc, manh])
  train_scores = [float(scores[i]) for i in training_ids]
  test_scores = [float(scores[i]) for i in testing_ids]
  X_train = np.array(training_in)
  Y_train = np.array(train_scores)
  reg = LinearRegression().fit(X_train, Y_train)
  X_test = np.array(testing_in)
  Y_test = np.array(test_scores)
  preds = reg.predict(X_test)
  
  m = mean_squared_error(Y_test, preds)
  mean += m
  print("Trial " + str(j) + "MSE: " + str(m))
print("Mean MSE:", mean / ntrials)
