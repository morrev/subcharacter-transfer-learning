from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import argparse
import torch
import csv
from wordsim_utils import *

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
    help = "finetuning dataset, either \"livedoor\" or \"wikipedia\"",
    default = "livedoor")
parser.add_argument("--num_trials",
    help = "number of trials to run for",
    default = "10")
parser.add_argument("--train_proportion",
    help = "proportion of dataset to be used for training, decimal format, 0 to 1",
    default = "0.8")
args = parser.parse_args()

# load JWSAN data
wstxtfile = 'data/jwsan-1400.csv'
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

# run linear regression
mean = 0
ntrials = int(args.num_trials)
for j in range(ntrials):
  training_in = []
  training_ids = random.sample([i for i in range(len(w1s))], int(len(w1s) * 0.8))
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
