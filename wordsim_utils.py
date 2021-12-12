import math
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def cos_sim(v1, v2):
  dt = torch.dot(v1, v2)
  return (dt / (torch.norm(v1) * torch.norm(v2))).item()

def euclidean(v1, v2):
  return math.sqrt(torch.sum((v1 - v2) ** 2))

def manhattan(v1, v2):
  return torch.sum(v1 - v2).item()
