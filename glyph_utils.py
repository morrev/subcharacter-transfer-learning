from transformers import AutoTokenizer
# from datasets.bert_dataset import BertDataset
# from models.modeling_glycebert import GlyceBertModel
from transformers.models.bert.modeling_bert import BertModel

CHINESEBERT_PATH = "~/ChineseBERT-large"
tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-char-v2")
# chinese_bert_tokenizer = BertDataset(CHINESEBERT_PATH)
# chinese_bert = GlyceBertModel.from_pretrained(CHINESEBERT_PATH)

def get_glyph_embeddings(sentence, pooled):
  if not pooled:
    # Tokenize accroding to japanese tokenizer
    tokens = tokenizer.tokenize(sentence)
    encoded = chinese_bert_tokenizer.tokenizer.encode_batch(tokens, add_special_tokens=False)
    g_embeddings = []
    for e in encoded:
      input_ids = torch.LongTensor([e.ids[0]]).view(1, -1)
      g = chinese_bert.embeddings.glyph_embeddings(input_ids)
      g = chinese_bert.embeddings.glyph_map(g).detach()
      g_embeddings.append(g)
    g_embeddings = torch.concat(g_embeddings).squeeze(dim=1)
    return g_embeddings
  encoded = chinese_bert_tokenizer.tokenizer.encode_batch(sentences, add_special_tokens=False)
  g_embeddings = []
  for e in encoded:
    input_ids = torch.LongTensor(e.ids).view(1, -1)
    g = chinese_bert.embeddings.glyph_embeddings(input_ids)
    g = chinese_bert.embeddings.glyph_map(g).detach()
    g_embeddings.append(g.mean(axis=1))
  return torch.cat(g_embeddings)

def text2glyph(X, padding=True, seq_length = 100):
  embs = []
  lens = []
  for sentence in X:
    emb_i = get_glyph_embeddings(sentence)
    embs.append(emb_i)
    lens.append(len(emb_i))

  max_len = max(lens)
  embs_padded = []
  for embs_i in embs:
    embs_padded.append(np.pad(embs_i, ((0, max_len-embs_i.shape[0]+1), (0, 0)), 'constant', constant_values=0))
  
  del(embs)
  gc.collect()

  return lens, np.array(embs_padded)
