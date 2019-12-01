import gpt_2_simple as gpt2
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM,BertForNextSentencePrediction
import tensorflow as tf
import torch
import nltk
nltk.download('punkt')

input_text = 'I still stay in bed.'
next_text = ''

sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess, checkpoint_dir='models/124M')

sents = []
for i in range(4):
    text = gpt2.generate(sess, return_as_list=True, checkpoint_dir='models/124M', length=200, prefix=input_text, truncate="<|endoftext|>")
    text = text[0]

    input_len = len(nltk.sent_tokenize(input_text))
    temp = nltk.sent_tokenize(text)
    sents += temp[input_len:-1]

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained model (weights)
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
model.eval()
predicts_text = []

for sent in sents:
    next_text = sent

    # Tokenized input
    text = "[CLS] " + input_text + " [SEP] " + next_text + " [SEP]"
    tokenized_text = tokenizer.tokenize(text)

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    len_1 = len(tokenizer.tokenize(input_text)) + 2 # [CLS] & [SEP]
    len_2 = len(tokenizer.tokenize(next_text)) + 1  # [SEP]
    segments_ids = len_1 * [0] + len_2 * [1]

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Predict is Next Sentence ?
    predictions = model(tokens_tensor, segments_tensors )

    predicts_text.append((predictions[0][0].item(), next_text))

final_shuang = sorted(predicts_text, key=lambda x: x[0], reverse=True)
if len(predicts_text) < 3:
    guess1 = "If I were you, maybe: " + final_shuang[0][1]
    guess2 = "Then maybe: " + final_shuang[1][1]
    robot_final_say = "God! It's really hard to guess what you humanbeings are thinking."
else:
    guess1 = "If I were you, maybe: " + final_shuang[0][1]
    guess2 = "Then maybe: " + final_shuang[1][1]
    guess3 = "Oh, really? or maybe: " + final_shuang[2][1]
    robot_final_say = "God! It's really hard to guess what you humanbeings are thinking."
    print(guess1)
    print(guess2)
    print(guess3)
