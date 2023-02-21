"""
    Plan-And-Write Style Autoregressive Model Baseline for Lexically-Constrained Text Generation
"""
import torch
import pickle
import torch.utils.data as datautils
from IPython import embed
import argparse
import os
import tqdm
from data.dataloader import NaiveTokenizer, RandomKeywordSequentialDataset, GivenKeywordSequentialDataset, DomainAdaptationSequentialDataset
from transformers.optimization import AdamW
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from models.decoder.modeling_cdgpt import CDGPT2LMHeadModel


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--continue_training",
        default=False,
        type=str2bool,
        required=False,
        help="Continue the training or start from scratch.",
    )
    parser.add_argument(
        "--dataset",
        default="yelp_review",
        type=str,
        required=False,
        help="Target dataset. Default: wmt16roen."
    )
    parser.add_argument(
        "--eval_mode",
        default=False,
        type=str2bool,
        required=False,
        help="load the latest checkpoint and run the eval pipeline.",
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        required=False,
        help="effective batchsize",
    )
    parser.add_argument(
        "--iter_per",
        default=8,
        type=int,
        required=False,
        help="cumulative gradient iteration cycle",
    )

    args = parser.parse_args()
    dataset_name = args.dataset

    if dataset_name[0:11] == "yelp_review":
        keyword_num = 7
    elif dataset_name[0:4] == "news":
        keyword_num = 4
    else:
        keyword_num = None

    eval_mode = args.eval_mode
    batch_size = args.batch_size
    iter_per = args.iter_per
    directory_identifier = "basemodel_%s_%s" % (dataset_name, "wlex")
    continue_training = args.continue_training
    if not os.path.exists("./checkpoints"):
        os.mkdir("checkpoints")
    if not os.path.exists("./checkpoints/%s" % directory_identifier):
        os.mkdir("checkpoints/%s" % directory_identifier)
    ckpt_name = "gpt2"

    try:
        tokenizer, dataset, val_dataset = pickle.load(
            open("checkpoints/baselinedataset-%s-%s.pyc" % (dataset_name, "wlex" if args.pseudo_seq2seq else "nolex"), "rb"))
        # raise NotImplementedError()
    except:
        tokenizer = GPT2Tokenizer.from_pretrained(ckpt_name)
        tokenizer.bos_token = "$"
        tokenizer.sep_token = "#"
        if args.pseudo_seq2seq:
            if dataset_name[0:11] == "yelp_review" or dataset_name[0:4] == "news":
                dataset = RandomKeywordSequentialDataset(tokenizer=tokenizer, max_len=384, keyword_num=keyword_num)
                dataset.add(dataset_name)
            else:
                import datasets
                dataset_raw = datasets.load_dataset(dataset_name, split="train")
                dataset = GivenKeywordSequentialDataset(tokenizer=tokenizer, max_len=384)
                dataset.add(dataset_raw, field_keywords="concepts", field_sequence="target")
        else:
            if dataset_name[0:11] == "yelp_review" or dataset_name[0:4] == "news":
                dataset = DomainAdaptationSequentialDataset(tokenizer=tokenizer, max_len=384)
                dataset.add(dataset_name)
            elif dataset_name == "common_gen":
                import datasets
                dataset_raw = datasets.load_dataset(dataset_name, split="train")
                dataset = DomainAdaptationSequentialDataset(tokenizer=tokenizer, max_len=384)
                dataset.add_huggingface(dataset_raw, field_keywords="concepts", field_sequence="target")
        val_dataset = None
        pickle.dump((tokenizer, dataset, val_dataset),
                    open("checkpoints/baselinedataset-%s-%s.pyc" % (dataset_name, "wlex" if args.pseudo_seq2seq else "nolex"), "wb"))

    dataset.__getitem__(0)
    base_model = GPT2LMHeadModel.from_pretrained(ckpt_name)
    base_model.train()
    base_model.cuda()
    generator = base_model

    dataloader = datautils.DataLoader(
        dataset, batch_size=batch_size // iter_per, shuffle=True, drop_last=False, pin_memory=True,
        num_workers=8
    )


    from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, get_constant_schedule
    opt = AdamW(lr=1e-5, weight_decay=0.02,
                eps=1e-8, params=base_model.parameters())
    # lr_scheduler = get_constant_schedule(opt)
    lr_scheduler = get_linear_schedule_with_warmup(opt, num_training_steps=5000000,
                                                   num_warmup_steps=400)
    if continue_training:
        generator.load_state_dict(torch.load("checkpoints/%s/pretrained" % directory_identifier))
        opt_ = torch.load("checkpoints/%s/opt" % directory_identifier)
        opt.load_state_dict(opt_)
        epoch_idx, iter_count = torch.load("checkpoints/%s/EpochIdx" % directory_identifier)
    else:
        epoch_idx, iter_count = 0, 0
        if type(dataloader) is list:
            for curriculum_step_i in range(len(dataloader)):
                flog_train = open("checkpoints/%s/log-%d.txt" % (directory_identifier, curriculum_step_i), "w")
                flog_train.close()
        flog_eval = open("checkpoints/%s/log-eval.txt" % (directory_identifier), "w")
        flog_eval.close()
    if eval_mode:
        epoch_id_spec = 4
        generator.load_state_dict(torch.load("checkpoints/%s/pretrained-%d" % (directory_identifier, epoch_id_spec)))
        torch.save(base_model.state_dict(), "checkpoints/%s/pretrained" % (directory_identifier))
        generator.eval()
        def get_top_k_and_prob(line, always_tracing=None, k=3):
            with torch.no_grad():
                input_ids = torch.tensor(
                    [tokenizer.bos_token_id] + tokenizer.encode("kid room dance") + [tokenizer.sep_token_id] + tokenizer.encode(line))
                logits = base_model(input_ids.cuda().unsqueeze(dim=0)).logits.log_softmax(dim=-1)
                logits = logits[0][-1]
                argmax_logits = logits.topk(k=k)
                print("Top %d token: %s" % (k, " ".join([tokenizer.decode(idx) for idx in argmax_logits.indices])))
                print("Top %d token: %s" % (k, " ".join(["%d" % idx for idx in argmax_logits.indices])))
                print("Top %d prob: %s"% (k, " ".join(["%f" % probs.exp() for probs in argmax_logits.values])))
                argmax_logits = (-logits).topk(k=k)
                print("Lowest %d token: %s" % (k, " ".join([tokenizer.decode(idx) for idx in argmax_logits.indices])))
                print("Lowest %d prob: %s"% (k, " ".join(["%f" % -probs for probs in argmax_logits.values])))
                if always_tracing is not None:
                    always_tracing = torch.tensor(tokenizer.encode(always_tracing, add_special_tokens=False))
                    print("Pinned token: %s" % (" ".join([tokenizer.decode(idx) for idx in always_tracing])))
                    print("Pinned token: %s" % (" ".join(["%d" % idx for idx in always_tracing])))
                    print("Pinned token prob: %s"% (" ".join(["%f" % probs.exp() for probs in logits[always_tracing]])))
        if type(tokenizer) is NaiveTokenizer:
            tokenizer.close_vocab()
        fout = open("./data/%s/%s-generated-baseline.txt" % (dataset_name, dataset_name), "w")
        with open("./data/%s/%s-keys.txt" % (dataset_name, dataset_name), "r") as fin:
            for line in tqdm.tqdm(fin.readlines()):
                input_ids = torch.tensor([tokenizer.bos_token_id] + tokenizer.encode(line.strip()) + [tokenizer.sep_token_id])
                # generated = generator.generate(input_ids=input_ids.cuda().unsqueeze(dim=0), max_length=300, do_sample=True, top_p=0.4, top_k=5)
                generated = generator.generate(input_ids=input_ids.cuda().unsqueeze(dim=0), max_length=300,
                                              num_beams=20, pad_token_id=tokenizer.eos_token_id)
                generated_str = tokenizer.decode(generated[0][len(input_ids):-1])
                print(generated_str, file=fout)
        fout.close()
        exit()
    truncate_num = 0
    for epoch_id in range(500):
        iterator = tqdm.tqdm(dataloader)
        for input_ids, mask, effective_len in iterator:
            max_len = effective_len.max()
            input_ids = input_ids.cuda()[:, truncate_num:max_len]
            mask = mask.cuda()[:, truncate_num:max_len - 1]

            logits = base_model(input_ids).logits.log_softmax(dim=-1)[:, :-1, :]
            nll_all = - (logits.gather(dim=-1, index=input_ids[:, 1:].unsqueeze(dim=-1)).reshape_as(mask) * mask)
            nll_reduced = nll_all.sum(dim=-1)

            if iter_count % iter_per == 0:
                opt.zero_grad()
            (nll_reduced / iter_per).mean(dim=0).backward()
            if iter_count % iter_per == iter_per - 1:
                opt.step()
                lr_scheduler.step()
                if (iter_count // iter_per) % 10 == 0:
                    iterator.write("Iteration %d-%d, Loss %f" % (
                    epoch_idx, iter_count // iter_per, nll_reduced.mean(dim=0).cpu().item()))
            # embed(); exit()
            iter_count += 1
        epoch_idx += 1
        torch.save(base_model.state_dict(), "checkpoints/%s/pretrained-%d" % (directory_identifier, epoch_id))
        torch.save((opt.state_dict(), lr_scheduler.state_dict()), "checkpoints/%s/opt" % directory_identifier)
        torch.save((epoch_idx, iter_count), "checkpoints/%s/EpochIdx" % directory_identifier)

if __name__ == "__main__":
    main()
