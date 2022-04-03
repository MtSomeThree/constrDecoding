from GPT2base import ConstrainedLM
from constraints import LogicalConstraintFunction
from neural_constr import NeuralConstraintFunction
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model, GPT2Config
from torch.optim import AdamW, Adam
from torch.nn.functional import log_softmax, softmax
import datasets
import copy
import torch
import argparse
import os

def sample_from_GPT2(model, tokenizer, constraint_function, args):
	if os.path.exists(args.samples_file):
		print ("Load Sampling Data from %s..."%(args.samples_file))
		# samples_list, labels_list, masks_list = torch.load(args.samples_file)
		# samples_list = [x.to(args.device) for x in samples_list]
		# labels_list = [x.to(args.device) for x in labels_list]
		# masks_list = [x.to(args.device) for x in masks_list]
		return torch.load(args.samples_file, map_location=args.device)

	print ("Initializing Sampling...")
	model.set_constraint_factor(0.0)

	sentence_prefix = ["I"] * args.sample_batch_size
	encodings_dict = tokenizer.batch_encode_plus(sentence_prefix)

	input_ids = torch.tensor(encodings_dict['input_ids']).to(args.device)
	attention_mask = torch.tensor(encodings_dict['attention_mask']).to(args.device)

	'''
	input_ids = tokenizer.encode(
		input_ids=input_ids,
		attention_mask=attention_mask,
		add_special_tokens=False,
		return_tensors="pt",
	).to(args.device)
	'''

	labels_list = []
	samples_list = []
	masks_list = []
	logprobs_list = []

	print("Sampling Data...")

	for i in range(args.num_sample_batches):
		outputs = model.generate(
			input_ids=input_ids,
			attention_mask=attention_mask,
			do_sample=True,
			max_length=args.max_length,  # desired output sentence length
			pad_token_id=model.config.eos_token_id,
			output_scores=True,
			return_dict_in_generate=True,
		)

		output_ids = outputs.sequences
		scores = outputs.scores

		labels = []
		masks = []
		logprobs = []
		texts = []

		for j in range(args.sample_batch_size):
			length = args.max_length - torch.where(output_ids[j] == model.config.eos_token_id)[0].shape[0]
			constr_input = tokenizer.decode(output_ids[j], skip_special_tokens=True)
			if isinstance(constraint_function, LogicalConstraintFunction):
				if constraint_function(constr_input):
					labels.append(1)
				else:
					labels.append(0)
			else:
				texts.append(constr_input)
			mask = [1] * length + [0] * (args.max_length - length)
			masks.append(mask)

			#logprob = outputs.sequences_scores[j] * float(length)

			
			logprob = 0.0
			for k in range(length - 1):
				logprob += log_softmax(scores[k][j], dim=0)[output_ids[j][k + 1]]
			
			logprobs.append(logprob)

		labels = torch.Tensor(labels).unsqueeze(1).to(args.device)
		if isinstance(constraint_function, NeuralConstraintFunction):
			_, labels, _ = constraint_function(texts, return_logits=True)
			labels.float().squeeze().unsqueeze(1).to(args.device)

		masks = torch.Tensor(masks).to(args.device)
		logprobs = torch.Tensor(logprobs).to(args.device)

		for j in range(int(args.sample_batch_size / args.batch_size)):
			labels_list.append(labels[j * args.batch_size : (j + 1) * args.batch_size])
			samples_list.append(output_ids[j * args.batch_size : (j + 1) * args.batch_size])
			masks_list.append(masks[j * args.batch_size : (j + 1) * args.batch_size])
			logprobs_list.append(logprobs[j * args.batch_size : (j + 1) * args.batch_size])
		if i % 10 == 9:
			print ("%d sample batches..."%(i + 1))

	# print ("Last batch sampled text:")
	# for j in range(args.sample_batch_size):
	# 	generated_text = tokenizer.decode(
	# 			output_ids[j],
	# 			clean_up_tokenization_spaces=True)
	# 	print (generated_text)
	torch.save((samples_list, labels_list, masks_list, logprobs_list), args.samples_file)
	return samples_list, labels_list, masks_list, logprobs_list

def train_rc(model, samples_list, labels_list, masks_list, logprobs_list, args):
	print ("Strat Training...")
	for p in model.parameters():
		p.requires_grad = False
	rc_parameters = []
	for n, p in model.named_parameters():
		if "model_rc" in n:
			p.requires_grad = True
			rc_parameters.append(p)

	print ("%d parameters in total"%(sum(p.numel() for p in model.parameters())))
	print ("%d parameters in rc"%(sum(p.numel() for p in model.parameters() if p.requires_grad)))

	optimizer = AdamW(params=rc_parameters, lr=args.lr)


	for epoch in range(args.num_epochs):
		cnt = 1
		loss_list = []
		for samples, labels, masks, logprobs in zip(samples_list, labels_list, masks_list, logprobs_list):
			labels = labels.float()
			#probs = softmax(logprobs, dim=0) * float(labels.shape[0])

			outputs = model(input_ids=samples, attention_mask=masks, rc_weights=logprobs, rc_labels=labels)
			loss = outputs.loss

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			cnt += 1
			loss_list.append(loss.item())

		print ("Epoch %d: avg loss: %.4f"%(epoch, torch.Tensor(loss_list).mean()))

		satisfied = test_rc(model, tokenizer, constraint_function, args, use_constr=True, sample_text=(epoch == args.num_epochs - 1))
		print (float(satisfied) / float(args.num_test) / float(args.sample_batch_size))


def test_rc(model, tokenizer, constraint_function, args, use_constr=True, sample_text=False):
	if use_constr:
		model.set_constraint_factor(1.0)
	else:
		model.set_constraint_factor(0.0)
	model.set_temperature(1.0)
	sentence_prefix = ["I"] * args.sample_batch_size
	encodings_dict = tokenizer.batch_encode_plus(sentence_prefix)

	input_ids = torch.tensor(encodings_dict['input_ids']).to(args.device)
	attention_mask = torch.tensor(encodings_dict['attention_mask']).to(args.device)

	#model(input_ids=input_ids)
	satisfied = 0

	for i in range(args.num_test):
		output_ids = model.generate(
			input_ids=input_ids,
			attention_mask=attention_mask,
			do_sample=True,
			max_length=args.max_length,  # desired output sentence length
			pad_token_id=model.config.eos_token_id,
		)

		if isinstance(constraint_function, NeuralConstraintFunction):
			texts = []
			for j in range(args.sample_batch_size):
				texts.append(tokenizer.decode(output_ids[j], skip_special_tokens=True))
			satisfied += constraint_function(texts)
		
		else:
			for j in range(args.sample_batch_size):
				if constraint_function(output_ids[j]):
					satisfied += 1

	if sample_text:
		for j in range(min(20, args.sample_batch_size)):
			generate_text = tokenizer.decode(output_ids[j], skip_special_tokens=True)
			print (generate_text)
			if isinstance(constraint_function, NeuralConstraintFunction):
				print (constraint_function(generate_text))
			else:
				print (constraint_function(output_ids[j]))

	model.set_temperature(args.temperature)
	return satisfied

def fine_tune_GPT2_with_pos_samples(model, samples_list, labels_list, masks_list, logprobs_list, args):
	model.set_constraint_factor(0.0)

	fine_tune_parameters = []
	for n, p in model.named_parameters():
		if "model_rc" in n:
			p.requires_grad = False
		else:
			fine_tune_parameters.append(p)

	optimizer = Adam(params=fine_tune_parameters, lr=args.lr)


	for epoch in range(args.num_epochs):
		cnt = 1
		loss_list = []
		for samples, labels, masks, logprobs in zip(samples_list, labels_list, masks_list, logprobs_list):
			labels = labels.float()
			if labels.sum() < 0.5:
				continue
			outputs = model(input_ids=samples, attention_mask=masks, labels=samples, rc_weights=logprobs, rc_labels=labels.squeeze(1))
			loss = outputs.loss

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			cnt += 1
			loss_list.append(loss.item())

		print ("Epoch %d: avg loss: %.4f"%(epoch, torch.Tensor(loss_list).mean()))

		satisfied = test_rc(model, tokenizer, constraint_function, args, use_constr=True, sample_text=(epoch == args.num_epochs - 1))
		print (float(satisfied) / float(args.num_test) / float(args.sample_batch_size))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--num_sample_batches', type=int, default=200)
	parser.add_argument('--sample_batch_size', type=int, default=512)
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--num_epochs', type=int, default=10)
	parser.add_argument('--lr', type=float, default=0.00003)
	parser.add_argument('--cuda', type=int, default=-1)
	parser.add_argument('--num_test', type=int, default=100)
	parser.add_argument('--max_length', type=int, default=30)
	parser.add_argument('--device', type=str, default=None)
	parser.add_argument('--dump_dir', type=str, default=None)
	parser.add_argument('--load_dir', type=str, default=None)
	parser.add_argument('--use_rc_transformer', action='store_true')
	parser.add_argument('--num_rc_layers', type=int, default=-1)
	parser.add_argument('--baseline_fine_tune', action='store_true')
	parser.add_argument('--constraint_id', type=int, default=1)
	parser.add_argument('--samples_file', type=str, default=None)
	parser.add_argument('--temperature', type=float, default=1.0)

	args = parser.parse_args()

	#constraint_function = LogicalConstraintFunction(args.constraint_id)
	constraint_function = NeuralConstraintFunction()
	constraint_function.init_formality()	

	if args.device is None:
		if args.cuda == -1:
			args.device = 'cpu'
		else:
			args.device = "cuda:%d"%(args.cuda)

	if args.samples_file is None:
		args.samples_file = './dump/formality_%d-%d-%d.pt'%(args.num_sample_batches, args.sample_batch_size, args.batch_size)

	model = ConstrainedLM.from_pretrained("gpt2")
	model.set_temperature(args.temperature)
	if args.num_rc_layers != -1:
		new_config = copy.copy(model.config)
		new_config.n_layer = args.num_rc_layers
		model.set_model_rc_transformer(GPT2Model(new_config))
	tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


	if args.baseline_fine_tune:

		model.to(args.device)
		samples_list, labels_list, masks_list, logprobs_list = sample_from_GPT2(model, tokenizer, constraint_function, args)
		fine_tune_GPT2_with_pos_samples(model, samples_list, labels_list, masks_list, logprobs_list, args)
	
	else:

		model.set_use_rc_transformer(args.use_rc_transformer)

		if args.load_dir is not None:
			model.model_rc.load_state_dict(torch.load(args.load_dir))
			model.to(args.device)
			satisfied = test_rc(model, tokenizer, constraint_function, args, use_constr=False, sample_text=False)
			print (float(satisfied) / float(args.num_test) / float(args.sample_batch_size))
		else:
			model.to(args.device)
			satisfied = test_rc(model, tokenizer, constraint_function, args, use_constr=False, sample_text=False)
			print (float(satisfied) / float(args.num_test) / float(args.sample_batch_size))
			samples_list, labels_list, masks_list, logprobs_list = sample_from_GPT2(model, tokenizer, constraint_function, args)
			model.set_constraint_factor(1.0)
			train_rc(model, samples_list, labels_list, masks_list, logprobs_list, args)

		satisfied = test_rc(model, tokenizer, constraint_function, args, use_constr=True, sample_text=True)

		print (float(satisfied) / float(args.num_test) / float(args.sample_batch_size))
		if args.dump_dir is None:
			args.dump_dir = './dump/%d.pt'%(int(float(10000 * satisfied) / float(args.num_test) / float(args.sample_batch_size)))
		if args.load_dir is None:
			torch.save(model.model_rc.state_dict(), args.dump_dir)

	

