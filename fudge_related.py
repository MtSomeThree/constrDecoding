import pickle
import os
import math
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline, set_seed, GPT2Tokenizer, GPT2Model, MarianTokenizer, MarianMTModel

from constant import *

class Model(nn.Module):
	def __init__(self, args, gpt_pad_id, vocab_size, rhyme_group_size=None, glove_embeddings=None, verbose=True):
		super(Model, self).__init__()
		self.topic = args.task == 'topic'
		self.formality = args.task == 'formality'
		self.iambic = args.task == 'iambic'
		self.rhyme = args.task == 'rhyme'
		self.newline = args.task == 'newline'

		self.marian_embed = nn.Embedding(gpt_pad_id + 1, HIDDEN_DIM, padding_idx=0) # 0 in marian is ''
		self.rnn = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, num_layers=3, bidirectional=False, dropout=0.5) # want it to be causal so we can learn all positions
		self.out_linear = nn.Linear(HIDDEN_DIM, 1)

	def forward(self, inputs, lengths=None, future_words=None, log_probs=None, syllables_to_go=None, future_word_num_syllables=None, rhyme_group_index=None, run_classifier=False):
		"""
		inputs: token ids, batch x seq, right-padded with 0s
		lengths: lengths of inputs; batch
		future_words: batch x N words to check if not predict next token, else batch
		log_probs: N
		syllables_to_go: batch
		"""
	  
		inputs = self.marian_embed(inputs)
		inputs = pack_padded_sequence(inputs.permute(1, 0, 2), lengths.cpu(), enforce_sorted=False)
		rnn_output, _ = self.rnn(inputs)
		rnn_output, _ = pad_packed_sequence(rnn_output)
		rnn_output = rnn_output.permute(1, 0, 2) # batch x seq x 300
		return self.out_linear(rnn_output).squeeze(2)


def avg_formality(preds, model, tokenizer, device='cuda:2'):
	probs = []
	for sent in preds:
		encoded_input = tokenizer.encode(sent, return_tensors='pt').to(device)
		lengths = torch.LongTensor([encoded_input.shape[1]]).to(device)
		scores = model(encoded_input, lengths=lengths) # batch x seq
		score = scores.flatten()[-1].item()
		probs.append(math.exp(score) / (1 + math.exp(score))) # sigmoided score = prob
	return np.mean(probs)

if __name__ == '__main__':


	device = 'cuda:2'
	tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-es-en')
	tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
	pad_id = tokenizer.encode(PAD_TOKEN)[0]

	checkpoint = torch.load('model.pth.tar', map_location=device)
	model_args = checkpoint['args']
	conditioning_model = Model(model_args, pad_id, 0) # no need to get the glove embeddings when reloading since they're saved in model ckpt anyway
	conditioning_model.load_state_dict(checkpoint['state_dict'])
	conditioning_model = conditioning_model.to(device)
	conditioning_model.eval()

	pred = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']

	print('avg formality prob according to model', avg_formality(pred, conditioning_model, tokenizer, device=device))