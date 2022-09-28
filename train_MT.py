from torch.optim import AdamW, Adam
from torch.nn.functional import log_softmax, softmax
from transformers import MarianMTModel, MarianTokenizer
from marianMT import ConstrainedMT
from neural_constr import NeuralConstraintFunction
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
import datasets
import copy
import torch
import numpy as np
import sacrebleu
import argparse
import os


def sample_from_marianMT(model, tokenizer, source_texts, constraint_function, args):
    print (args.samples_file)
    if os.path.exists(args.samples_file):
        print ("Loading Sampling Data from %s..."%(args.samples_file))
        return torch.load(args.samples_file, map_location='cpu')

    print ("Initializing Sampling...")
    model.set_constraint_factor(0.0)
    model.set_temperature(args.temperature)
    model.to(args.device)

    inputs_list = []
    labels_list = []
    samples_list = []
    d_masks_list = []
    e_masks_list = []
    logprobs_list = []

    print("Sampling Data...")

    cnt = 0

    cur_max_length = 1

    length = len(source_texts)

    for idx in tqdm(range(length)):
        source_text = source_texts[idx]
        cnt += 1

        texts = [source_text] * args.sample_batch_size

        encodings_dict = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = torch.tensor(encodings_dict['input_ids']).to(args.device)
        attention_mask = torch.tensor(encodings_dict['attention_mask']).to(args.device)

        outputs = model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            do_sample=True, 
            output_scores=True, 
            return_dict_in_generate=True,
        )
        output_ids = outputs.sequences.to('cpu')

        labels = []
        d_masks = []
        e_masks = []
        logprobs = []
        for j in range(args.sample_batch_size):
            length = output_ids.shape[1] - torch.where(output_ids[j] == model.config.pad_token_id)[0].shape[0] + 1
            translated = tokenizer.decode(output_ids[j], skip_special_token=True)
            if constraint_function(translated):
                labels.append(1)
            else:
                labels.append(0)
            d_mask = [1] * length + [0] * (output_ids.shape[1] - length)
            d_masks.append(d_mask)

            logprob = outputs.sequences_scores[j] * float(length)
            logprobs.append(logprob)

        input_ids = input_ids.to('cpu')
        attention_mask = attention_mask.to('cpu')
        labels = torch.Tensor(labels).unsqueeze(1).to('cpu')
        d_masks = torch.Tensor(d_masks).to('cpu')
        logprobs = torch.Tensor(logprobs).to('cpu')

        for j in range(int(args.sample_batch_size / args.batch_size)):
            inputs_list.append(input_ids[j * args.batch_size : (j + 1) * args.batch_size])
            labels_list.append(labels[j * args.batch_size : (j + 1) * args.batch_size])
            samples_list.append(output_ids[j * args.batch_size : (j + 1) * args.batch_size])
            d_masks_list.append(d_masks[j * args.batch_size : (j + 1) * args.batch_size])
            e_masks_list.append(attention_mask[j * args.batch_size : (j + 1) * args.batch_size])
            logprobs_list.append(logprobs[j * args.batch_size : (j + 1) * args.batch_size])

    # print ("Last batch sampled text:")
    # for j in range(args.sample_batch_size):
    #    generated_text = tokenizer.decode(
    #            output_ids[j],
    #            clean_up_tokenization_spaces=True)
    #    print (generated_text)
    torch.save((inputs_list, samples_list, labels_list, d_masks_list, e_masks_list, logprobs_list), args.samples_file)
    return (inputs_list, samples_list, labels_list, d_masks_list, e_masks_list, logprobs_list)

def sample_from_marianMT2(model, tokenizer, source_texts, constraint_function, args):
    if os.path.exists(args.samples_file):
        print ("Loading Sampling Data from %s..."%(args.samples_file))
        return torch.load(args.samples_file, map_location='cpu')

    print ("Initializing Sampling...")
    model.set_constraint_factor(0.0)
    model.set_temperature(args.temperature)
    model.to(args.device)

    inputs_list = []
    labels_list = []
    samples_list = []
    d_masks_list = []
    e_masks_list = []
    logprobs_list = []

    print("Sampling Data...")

    cnt = 0

    cur_max_length = 1

    corpus_length = len(source_texts)

    batch_size = 32
    
    for epoch in range(args.sample_batch_size):
        for idx in tqdm(range(int(corpus_length / batch_size + 1))):
            source_text = source_texts[idx * batch_size: min(corpus_length, (idx + 1) * batch_size)]
            cnt += 1

            texts = source_text
            encodings_dict = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            input_ids = torch.tensor(encodings_dict['input_ids']).to(args.device)
            attention_mask = torch.tensor(encodings_dict['attention_mask']).to(args.device)

            outputs = model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                do_sample=True, 
                output_scores=True, 
                return_dict_in_generate=True,
            )
            output_ids = outputs.sequences.to('cpu')

            labels = []
            d_masks = []
            e_masks = []
            logprobs = []
            for j in range(args.sample_batch_size):
                length = output_ids.shape[1] - torch.where(output_ids[j] == model.config.pad_token_id)[0].shape[0] + 1
                translated = tokenizer.decode(output_ids[j], skip_special_token=True)
                if constraint_function(translated):
                    labels.append(1)
                else:
                    labels.append(0)
                d_mask = [1] * length + [0] * (output_ids.shape[1] - length)
                d_masks.append(d_mask)

                logprob = outputs.sequences_scores[j] * float(length)
                logprobs.append(logprob)

            input_ids = input_ids.to('cpu')
            attention_mask = attention_mask.to('cpu')
            labels = torch.Tensor(labels).unsqueeze(1).to('cpu')
            d_masks = torch.Tensor(d_masks).to('cpu')
            logprobs = torch.Tensor(logprobs).to('cpu')

            for j in range(int(args.sample_batch_size / args.batch_size)):
                inputs_list.append(input_ids[j * args.batch_size : (j + 1) * args.batch_size])
                labels_list.append(labels[j * args.batch_size : (j + 1) * args.batch_size])
                samples_list.append(output_ids[j * args.batch_size : (j + 1) * args.batch_size])
                d_masks_list.append(d_masks[j * args.batch_size : (j + 1) * args.batch_size])
                e_masks_list.append(attention_mask[j * args.batch_size : (j + 1) * args.batch_size])
                logprobs_list.append(logprobs[j * args.batch_size : (j + 1) * args.batch_size])

    # print ("Last batch sampled text:")
    # for j in range(args.sample_batch_size):
    #    generated_text = tokenizer.decode(
    #            output_ids[j],
    #            clean_up_tokenization_spaces=True)
    #    print (generated_text)
    torch.save((inputs_list, samples_list, labels_list, d_masks_list, e_masks_list, logprobs_list), args.samples_file)
    return (inputs_list, samples_list, labels_list, d_masks_list, e_masks_list, logprobs_list)


def train_rc(model, train_data, valid_texts, constraint_function, args, valid_ref=None):
    print ("Strat Training...")
    model.to(args.device)
    inputs_list, samples_list, labels_list, d_masks_list, e_masks_list, logprobs_list = train_data
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
    train_length = len(inputs_list)

    max_score = -1

    if args.warm_start:
        print ("Start Warm Start:")
        for epoch in range(2):
            for idx in tqdm(range(train_length)):
                input_ids = inputs_list[idx].to(args.device)
                samples = samples_list[idx].to(args.device)
                d_masks = d_masks_list[idx].to(args.device)
                e_masks = e_masks_list[idx].to(args.device)
                labels = labels_list[idx].float().to(args.device)
                logprobs = logprobs_list[idx].to(args.device)

                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=e_masks, 
                    decoder_input_ids=samples,
                    decoder_attention_mask=d_masks,
                    rc_weights=logprobs, 
                    rc_labels=labels,
                    fine_tune=True
                )
                loss = outputs.loss
                #print (loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        satisfied, bleu_score, translated = test_rc(model, tokenizer, constraint_function, valid_texts, args, use_constr=True, sample_text=False, references=valid_ref)
        if bleu_score > max_score:
            max_score = bleu_score
            best_translated = translated
    
    optimizer = AdamW(params=rc_parameters, lr=args.lr)

    for epoch in range(args.num_epochs):
        cnt = 1
        loss_list = []

        for idx in tqdm(range(train_length)):
            input_ids = inputs_list[idx].to(args.device)
            samples = samples_list[idx].to(args.device)
            d_masks = d_masks_list[idx].to(args.device)
            e_masks = e_masks_list[idx].to(args.device)
            labels = labels_list[idx].float().to(args.device)
            logprobs = logprobs_list[idx].to(args.device)

            outputs = model(
                input_ids=input_ids, 
                attention_mask=e_masks, 
                decoder_input_ids=samples,
                decoder_attention_mask=d_masks,
                rc_weights=logprobs, 
                rc_labels=labels
            )
            loss = outputs.loss
            #print (loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cnt += 1
            loss_list.append(loss.item())

        print ("Epoch %d: avg loss: %.4f"%(epoch, torch.Tensor(loss_list).mean()))

        satisfied, bleu_score, translated = test_rc(model, tokenizer, constraint_function, valid_texts, args, use_constr=(args.num_epochs != 1), sample_text=(epoch == args.num_epochs - 1), references=valid_ref)
        if bleu_score > max_score:
            max_score = bleu_score
            best_translated = translated
            model.save_rc(args.save_dir)


    with open(args.log, 'w') as f:
        for line in best_translated:
            f.write("%d %s\n"%(constraint_function(line), line))

def test_rc(model, tokenizer, constraint_function, source_texts, args, use_constr=True, sample_text=False, references=None):
    if use_constr:
        model.set_constraint_factor(1.0)
    else:
        model.set_constraint_factor(0.0)
    model.set_temperature(1.0)
    model.to(args.device)
    model.eval()

    #model(input_ids=input_ids)
    satisfied = 0
    satisfied_soft = 0.0

    if references is None:
        return_bleu = False
    else:
        return_bleu = True

    num_test = 0.0

    translated = []

    if args.test_mode != 'sample':
        test_batch_size = 16
        length = len(source_texts)
        iter_num = int(length / test_batch_size)

    for idx in tqdm(range(iter_num + 1)): # TODO: what if length % 32 == 0?
        if args.test_mode == 'sample':
            num_per_text = args.sample_batch_size
            texts = [source_text] * num_per_text
        else:
            num_per_text = 1
            texts = source_texts[idx * test_batch_size: min(length, (idx + 1) *  test_batch_size)]

        encodings_dict = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = torch.tensor(encodings_dict['input_ids']).to(args.device)
        attention_mask = torch.tensor(encodings_dict['attention_mask']).to(args.device)

        outputs = model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            do_sample=(args.test_mode == 'sample'), 
        )

        for output in outputs:
            t = tokenizer.decode(output, skip_special_tokens=True)
            translated.append(t)

    satisfied += constraint_function(translated)
    satisfied_soft += constraint_function(translated, soft=True)

    if sample_text:
        print (texts[-1])
        print (references[0][-1])
        print (references[1][-1])
        print (translated[-1])

    if args.test_mode == 'sample':
        total = float(args.sample_batch_size * len(source_texts))
    else:
        total = float(len(source_texts))

    print (float(satisfied) / total)
    print (float(satisfied_soft) / total)

    model.set_temperature(args.temperature)
    if return_bleu:
        score = sacrebleu.corpus_bleu(translated, references)
        print (score)
        return satisfied, score.score, translated
    return satisfied

# def fine_tune_GPT2_with_pos_samples(model, samples_list, labels_list, masks_list, logprobs_list, args):
#   model.set_constraint_factor(0.0)

#   fine_tune_parameters = []
#   for n, p in model.named_parameters():
#       if "model_rc" in n:
#           p.requires_grad = False
#       else:
#           fine_tune_parameters.append(p)

#   optimizer = Adam(params=fine_tune_parameters, lr=args.lr)


#   for epoch in range(args.num_epochs):
#       cnt = 1
#       loss_list = []
#       for samples, labels, masks, logprobs in tqdm(zip(samples_list, labels_list, masks_list, logprobs_list)):
#           labels = labels.float()
#           if labels.sum() < 0.5:
#               continue
#           outputs = model(input_ids=samples, attention_mask=masks, labels=samples, rc_weights=logprobs, rc_labels=labels.squeeze(1))
#           loss = outputs.loss

#           optimizer.zero_grad()
#           loss.backward()
#           optimizer.step()

#           cnt += 1
#           loss_list.append(loss.item())

#       print ("Epoch %d: avg loss: %.4f"%(epoch, torch.Tensor(loss_list).mean()))

#       satisfied, _ = test_rc(model, tokenizer, args, use_constr=True, sample_text=(epoch == args.num_epochs - 1))
#       print (float(satisfied) / float(args.num_test) / float(args.sample_batch_size))

def fine_tune_epoch(model, tokenizer, input_list, ref_list):
    model.set_constraint_factor(0.0)

    fine_tune_parameters = []
    for n, p in model.named_parameters():
        if "model_rc" in n:
            p.requires_grad = False
        else:
            fine_tune_parameters.append(p)

    optimizer = Adam(params=fine_tune_parameters, lr=args.lr)

    length = int(len(input_list) / args.batch_size)

    loss_list = []

    for idx in tqdm(range(length)):
        texts = input_list[idx * args.batch_size: (idx + 1) * args.batch_size]
        refs = ref_list[idx * args.batch_size: (idx + 1) * args.batch_size]

        encodings_dict = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = torch.tensor(encodings_dict['input_ids']).to(args.device)
        attention_mask = torch.tensor(encodings_dict['attention_mask']).to(args.device)

        encodings_dict = tokenizer(refs, return_tensors="pt", padding=True, truncation=True, max_length=128)
        decoder_ids = torch.tensor(encodings_dict['input_ids']).to(args.device)
        decoder_mask = torch.tensor(encodings_dict['attention_mask']).to(args.device)

        padding = torch.zeros(args.batch_size).long().unsqueeze(1).to(args.device) + tokenizer.pad_token_id
        decoder_ids = torch.cat((padding, decoder_ids), 1)
        mask_padding = torch.ones(args.batch_size).long().unsqueeze(1).to(args.device)
        decoder_mask = torch.cat((mask_padding, decoder_mask), 1)

        outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    decoder_input_ids=decoder_ids,
                    decoder_attention_mask=decoder_mask,
                    rc_weights=None, 
                    rc_labels=None,
                    fine_tune=True
                )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

    print ("avg loss: %.4f"%(torch.Tensor(loss_list).mean()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_batch_size', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.00003)
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--max_length', type=int, default=30)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--dump_dir', type=str, default=None)
    parser.add_argument('--load_dir', type=str, default=None)
    parser.add_argument('--baseline_fine_tune', action='store_true')
    parser.add_argument('--samples_file', type=str, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--regularization', type=float, default=0.5)
    parser.add_argument('--test_mode', type=str, default='greedy')
    parser.add_argument('--num_ref', type=int, default=2)
    parser.add_argument('--keyword', type=str, default='10k')
    parser.add_argument('--rc_layers', type=int, default=3)
    parser.add_argument('--warm_start', action='store_true')
    parser.add_argument('--zero_init', action='store_true')
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--log', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--reg_type', type=int, default=3)

    args = parser.parse_args()

    torch.manual_seed(9131217)

    train_texts = ["Me gusta tocar el piano.", "Esta es una tarea de traducción automática."]
    valid_texts = ["Me gusta tocar el piano.", "Esta es una tarea de traducción automática."]
    test_texts = ["Me gusta tocar el piano.", "Esta es una tarea de traducción automática."]

    if args.device is None:
        if args.cuda == -1:
            args.device = 'cpu'
        else:
            args.device = 'cuda:%d'%(args.cuda)

    constraint_function = NeuralConstraintFunction()
    constraint_function.init_FUDGE_formality()
    constraint_function.set_device(args.device)

    if args.samples_file is None:
        args.samples_file = './dump/MT/fisher_%s_%d-%d-%.2f.pt'%(args.keyword, 
            args.sample_batch_size, args.batch_size, args.temperature)
    if args.log is None:
        args.log = './dump/MT/fisher_%s_%d-%d-%.2f.log'%(args.keyword, 
            args.sample_batch_size, args.batch_size, args.temperature)

    if args.fine_tune:
        args.keyword = 'finetune'

    if args.save_dir is None:
        args.save_dir = './dump/reg_MT/%s_%d-%d-%d-%.2f.ckpt'%(args.keyword,
            args.sample_batch_size, args.batch_size, args.rc_layers, args.temperature)
        if args.warm_start:
            args.save_dir = './dump/MT/%s_%d-%d-%d-%.2f-warm.ckpt'%(args.keyword,
            args.sample_batch_size, args.batch_size, args.rc_layers, args.temperature)

    model_name = "Helsinki-NLP/opus-mt-es-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = ConstrainedMT.from_pretrained(model_name, rc_layers=args.rc_layers)
    model.set_regularization(args.regularization)
    model.set_reg_type(args.reg_type)
    if args.zero_init:
        model.zero_init_rc()

    # dataset = datasets.load_dataset("wmt14", "de-en")
    # valid = dataset['validation']['translation']
    # train = dataset['train']['translation']

    # valid_de = [x['de'] for x in valid]
    # valid_en = [x['en'] for x in valid]

    # # train_de = [x['de'] for x in train]
    # # train_en = [x['en'] for x in train]

    num_reference = args.num_ref

    valid_es = []
    train_es = []
    test_es = []
    train_en = []
    valid_en = [[]] * num_reference
    test_en = [[]] * num_reference

    with open('fisher-callhome-corpus/corpus/ldc/fisher_test.es', 'r') as f:
        for line in f:
            test_es.append(line.strip())

    with open('fisher-callhome-corpus/corpus/ldc/fisher_train.es', 'r') as f:
        for line in f:
            train_es.append(line.strip())

    with open('fisher-callhome-corpus/corpus/ldc/fisher_dev.es', 'r') as f:
        for line in f:
            valid_es.append(line.strip())

    with open('fisher-callhome-corpus/corpus/ldc/fisher_train.en', 'r') as f:
        for line in f:
            train_en.append(line.strip())

    for i in range(num_reference):
        with open('fluent-fisher/noids/dev.noid.cleaned_%d'%(i), 'r') as f:
            for line in f:
                #clean_line = line.strip().split()[1:]
                #clean_line = " ".join(clean_line)
                valid_en[i].append(line.strip())

    for i in range(num_reference):
        with open('fluent-fisher/noids/test.noid.cleaned_%d'%(i), 'r') as f:
            for line in f:
                #clean_line = line.strip().split()[1:]
                #clean_line = " ".join(clean_line)
                test_en[i].append(line.strip())

    if args.fine_tune:
        model.to(args.device)
        for i in range(3):
            fine_tune_epoch(model, tokenizer, train_es, train_en)
            test_rc(model, tokenizer, constraint_function, test_es, args, use_constr=False, sample_text=True, references=test_en)
            finetune_dump_dir = './dump/reg_MT/pretrained-%d.ckpt'%(i)


    if args.load_dir is not None:
        decoder_dict, linear_dict = torch.load(args.load_dir)
        model.model_rc_decoder.load_state_dict(decoder_dict)
        model.model_rc_linear.load_state_dict(linear_dict)
        model.to(args.device)

        satisfied, _, _ = test_rc(model, tokenizer, constraint_function, test_es, args, use_constr=True, sample_text=True, references=test_en)

    else:

        constraint_function.set_device(args.device)
        train_data = sample_from_marianMT(model, tokenizer, train_es, constraint_function, args)
        model.set_constraint_factor(0.0)
        satisfied, _, translated = test_rc(model, tokenizer, constraint_function, test_es, args, use_constr=True, sample_text=False, references=test_en)
        with open(args.log, "w") as f:
            for line in translated:
                f.write("%s\n"%(line))
        print ("Output base done!")
        train_rc(model, train_data, test_es, constraint_function, args, valid_ref=test_en)
        satisfied, _, _ = test_rc(model, tokenizer, constraint_function, test_es, args, use_constr=False, sample_text=True, references=test_en)

