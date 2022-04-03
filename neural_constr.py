from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import XLMRobertaTokenizerFast, XLMRobertaForSequenceClassification
from transformers import MarianTokenizer
from torch.optim import AdamW, Adam
from torch.nn.functional import one_hot
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence

from constant import *

import torch
import numpy as np
import random
import math
import os
import argparse

class FUDGEModel(torch.nn.Module):
	def __init__(self, args, gpt_pad_id, vocab_size, rhyme_group_size=None, glove_embeddings=None, verbose=True):
		super(FUDGEModel, self).__init__()
		self.topic = args.task == 'topic'
		self.formality = args.task == 'formality'
		self.iambic = args.task == 'iambic'
		self.rhyme = args.task == 'rhyme'
		self.newline = args.task == 'newline'

		self.marian_embed = torch.nn.Embedding(gpt_pad_id + 1, HIDDEN_DIM, padding_idx=0) # 0 in marian is ''
		self.rnn = torch.nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, num_layers=3, bidirectional=False, dropout=0.5) # want it to be causal so we can learn all positions
		self.out_linear = torch.nn.Linear(HIDDEN_DIM, 1)

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

class NeuralConstraintFunction(object):
	def __init__(self, model=None, tokenizer=None):
		self.model = model
		self.tokenizer = tokenizer
		self.device = 'cpu'
		self.batch_size = 512
		self.fudge = False

	def set_device(self, device):
		self.device = device
		self.model.to(device)

	def init_sentiment(self, dump_dir='./dump/sentiment.pt'):
		self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
		checkpoint = torch.load(dump_dir, map_location='cpu')
		self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
		self.model.load_state_dict(checkpoint['model_state_dict'])

	def init_formality(self, dump_dir='./dump/formality.pt'):
		self.tokenizer = XLMRobertaTokenizerFast.from_pretrained('SkolkovoInstitute/xlmr_formality_classifier')
		self.model = XLMRobertaForSequenceClassification.from_pretrained('SkolkovoInstitute/xlmr_formality_classifier')

	def init_GYAFC_formality(self, dump_dir='./dump/GYAFC_formality.pt'):
		self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
		checkpoint = torch.load(dump_dir, map_location='cpu')
		self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
		self.model.load_state_dict(checkpoint['model_state_dict'])

	def init_FUDGE_formality(self, dump_dir='./test_evaluator_gyafc_family_relationships/model.pth.tar'):
		self.fudge = True
		self.tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-es-en')
		self.tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})

		checkpoint = torch.load(dump_dir, map_location=self.device)
		model_args = checkpoint['args']
		pad_id = self.tokenizer.encode(PAD_TOKEN)[0]
		self.model = FUDGEModel(model_args, pad_id, 0) # no need to get the glove embeddings when reloading since they're saved in model ckpt anyway
		self.model.load_state_dict(checkpoint['state_dict'])
		self.model = self.model.to(self.device)
		self.model.eval()

	def avg_formality(preds):
		probs = []
		for sent in preds:
			encoded_input = self.tokenizer.encode(sent, return_tensors='pt').to(self.device)
			lengths = torch.LongTensor([encoded_input.shape[1]]).to(self.device)
			scores = self.model(encoded_input, lengths=lengths) # batch x seq
			score = scores.flatten()[-1].item()
			probs.append(math.exp(score) / (1 + math.exp(score))) # sigmoided score = prob
		return np.mean(probs)
		
	def __call__(self, text, return_logits=False, soft=False):
		if self.fudge:
			probs = []
			for sent in text:
				encoded_input = self.tokenizer.encode(sent, return_tensors='pt').to(self.device)
				lengths = torch.LongTensor([encoded_input.shape[1]]).to(self.device)
				scores = self.model(encoded_input, lengths=lengths) # batch x seq
				score = scores.flatten()[-1].item()
				if soft:
					probs.append(math.exp(score) / (1 + math.exp(score))) # sigmoided score = prob
				else:
					if score > 0:
						probs.append(1)
					else:
						probs.append(0)
			return np.sum(probs)

		encoding_dict = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
		input_ids = encoding_dict['input_ids'].to(self.device)
		attention_mask = encoding_dict['attention_mask'].to(self.device)

		outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

		pred_logits = outputs.logits
		pred = pred_logits.max(dim=1)[1]
		satisfied = int(pred.sum())

		if return_logits:
			return satisfied, pred, pred_logits
		else:
			return satisfied


def eval_constr_model(model, tokenizer, loader, args=None):
	if args is not None:
		device = args.device
	else:
		device = "cuda:6"

	model.to(device)
	total = 0
	correct = 0
	for samples, labels in loader:
		total += len(samples)
		labels = torch.LongTensor(labels).to(device)

		encoding_dict = tokenizer(samples, return_tensors='pt', padding=True, truncation=True, max_length=128)
		input_ids = encoding_dict['input_ids'].to(device)
		attention_mask = encoding_dict['attention_mask'].to(device)
		outputs = model(input_ids=input_ids, attention_mask=attention_mask, 
			labels=one_hot(labels, num_classes=2).float())

		pred_logits = outputs.logits
		pred = pred_logits.max(dim=1)[1]

		correct += torch.where(pred==labels)[0].shape[0]

	print ("Eval done! Accuracy: %.4f, Loss: %.4f"%(float(correct) / float(total), outputs.loss.item()))

	return total, correct, outputs.loss.item()


def train_constr_model(train_samples, train_labels, valid_samples, valid_labels, args=None):
	tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
	model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

	if args is not None:
		num_epochs = args.num_epochs
		lr = args.lr
		device = args.device
	else:
		num_epochs = 10
		lr = 0.00002
		device = "cuda:6"

	model.to(device)
	optimizer = AdamW(model.parameters(), lr=lr)

	for epoch in range(num_epochs):
		loss_list = []
		train_loader = dataset_loader(train_samples, train_labels)
		for samples, labels in train_loader:
			encoding_dict = tokenizer(samples, return_tensors='pt', padding=True, truncation=True, max_length=128)
			input_ids = encoding_dict['input_ids'].to(device)
			attention_mask = encoding_dict['attention_mask'].to(device)
			labels = one_hot(torch.LongTensor(labels), num_classes=2).float().to(device)
			outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

			optimizer.zero_grad()
			loss = outputs.loss
			loss.backward()
			optimizer.step()

			loss_list.append(loss.item())
		print ("Epoch %d, avg loss: %.4f"%(epoch, torch.Tensor(loss_list).mean()))
		valid_loader = dataset_loader(valid_samples, valid_labels)
		total, correct, loss = eval_constr_model(model, tokenizer, valid_loader)

	return model, tokenizer

def dataset_loader(samples, labels, batch_size=16):
	length = len(samples)
	
	shuffle_list = list(zip(samples, labels))
	random.shuffle(shuffle_list)
	samples, labels = zip(*shuffle_list)
	samples = list(samples)
	labels = list(labels)

	samples_batch = []
	labels_batch = []

	N = int(length / batch_size)

	for i in range(N):
		yield samples[i * batch_size: (i + 1) * batch_size], labels[i * batch_size: (i + 1) * batch_size]

	if batch_size * N < length:
		yield samples[N * batch_size:], labels[N * batch_size:]

def init_sentiment(dump_dir='./dump/sentiment.pt'):
	dataset = load_dataset("SetFit/sst2")
	if dump_dir is None or (not os.path.exists(dump_dir)):
		model, tokenizer = train_constr_model(dataset['train']['text'], dataset['train']['label'], 
						dataset['validation']['text'], dataset['validation']['label'])
		torch.save({'model_state_dict': model.state_dict()}, dump_dir)
	else:
		tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
		checkpoint = torch.load(dump_dir, map_location='cpu')
		model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
		model.load_state_dict(checkpoint['model_state_dict'])
		
	test_loader = dataset_loader(dataset['test']['text'], dataset['test']['label'])

	return model, tokenizer

def init_formality(dump_dir='./dump/formality.pt'):
	tokenizer = XLMRobertaTokenizerFast.from_pretrained('SkolkovoInstitute/xlmr_formality_classifier')
	model = XLMRobertaForSequenceClassification.from_pretrained('SkolkovoInstitute/xlmr_formality_classifier')

	return model, tokenizer

def load_text_list(file_dir, max_length=128):
	l = []
	with open(file_dir, 'r') as f:
		for line in f:
			if len(line) > max_length:
				l.append(line[:max_length - 1])
			else:
				l.append(line.strip())
	return l

def init_GYAFC_formality(dump_dir='./dump/GYAFC.pt'):
	if dump_dir is None or (not os.path.exists(dump_dir)):
		train_0 = load_text_list('Entertainment_Music/train/informal')
		train_1 = load_text_list('Entertainment_Music/train/formal')
		train_labels = [0] * len(train_0) + [1] * len(train_1)
		train_samples = train_0 + train_1
		valid_0 = load_text_list('Entertainment_Music/tune/informal')
		valid_1 = load_text_list('Entertainment_Music/tune/formal')
		valid_labels = [0] * len(valid_0) + [1] * len(valid_1)
		valid_samples = valid_0 + valid_1

		model, tokenizer = train_constr_model(train_samples, train_labels, 
			valid_samples, valid_labels)
		torch.save({'model_state_dict': model.state_dict()}, dump_dir)
	else:
		tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
		checkpoint = torch.load(dump_dir, map_location='cpu')
		model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
		model.load_state_dict(checkpoint['model_state_dict'])

	return model, tokenizer


def neural_constr_function(model, tokenizer, text, device='cpu'):
	model.to(device)

	encoding_dict = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
	input_ids = encoding_dict['input_ids'].to(device)
	attention_mask = encoding_dict['attention_mask'].to(device)

	outputs = model(input_ids=input_ids, attention_mask=attention_mask)

	pred_logits = outputs.logits
	pred = pred_logits.max(dim=1)[1]

	return pred, pred_logits



if __name__ == "__main__":
	fudge = NeuralConstraintFunction()
	fudge.init_FUDGE_formality()

	pred = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']
	print (fudge(pred) / 3.0, fudge(pred, soft=True) / 3.0)
