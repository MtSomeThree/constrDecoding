"""
    Plan-And-Write Style Autoregressive Model Baseline for Lexically-Constrained Text Generation
"""
import torch
import pickle
import torch.utils.data as datautils
from IPython import embed
import argparse
import os
import numpy as np
import tqdm
from data.dataloader import NaiveTokenizer, RandomKeywordSequentialDataset, GivenKeywordSequentialDataset, DomainAdaptationSequentialDataset
from transformers.optimization import AdamW
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from models.decoder.modeling_cdgpt import CDGPT2LMHeadModel
from data.dataloader import LexicalCheckingDataset, ExactFormLexicalCheckingDataset


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
        type=bool,
        required=False,
        help="load the latest checkpoint and run the eval pipeline.",
    )
    parser.add_argument(
        "--batch_size",
        default=256,
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
    parser.add_argument(
        "--bellman_reg",
        default="1.00",
        type=str,
        required=False,
        help="strength of the bellman regularization term",
    )


    args = parser.parse_args()
    dataset_name = args.dataset
    keyword_num = 7 if dataset_name == "yelp_review" else 4
    eval_mode = args.eval_mode
    batch_size = args.batch_size
    iter_per = args.iter_per
    directory_identifier_raw = "basemodel_%s_%s" % (dataset_name, "nolex"
    if (dataset_name[0:11] == "yelp_review" or dataset_name[0:4] == "news") else "wlex")
    directory_identifier = "baseline_randkey_%s" % dataset_name
    continue_training = args.continue_training
    if not os.path.exists("./checkpoints"):
        os.mkdir("checkpoints")
    if not os.path.exists("./checkpoints/%s" % directory_identifier):
        os.mkdir("checkpoints/%s" % directory_identifier)
    try:
        tokenizer, dataset, val_dataset = pickle.load(
            open("checkpoints/baselinedataset-%s.pyc" % (dataset_name), "rb"))
    except:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
        tokenizer.bos_token = "$"
        tokenizer.sep_token = "#"
        if dataset_name[0:11] == "yelp_review" or dataset_name[0:4] == "news":
            dataset = RandomKeywordSequentialDataset(tokenizer=tokenizer, max_len=384, keyword_num=keyword_num)
            dataset.add(dataset_name)
        else:
            import datasets
            dataset_raw = datasets.load_dataset(dataset_name, split="train")
            dataset = GivenKeywordSequentialDataset(tokenizer=tokenizer, max_len=384)
            dataset.add(dataset_raw, field_keywords="concepts", field_sequence="target")
        val_dataset = None
        pickle.dump((tokenizer, dataset, val_dataset), open("checkpoints/baselinedataset-%s.pyc" % (dataset_name), "wb"))

    dataset.__getitem__(0)
    base_model = GPT2LMHeadModel.from_pretrained("gpt2-large")
    base_model.eval()
    base_model.cuda()
    config = GPT2Config.from_pretrained("gpt2", # much smaller model
        n_layer=4,
    )
    generator = CDGPT2LMHeadModel(config=config, base_model=base_model)
    generator.cuda()
    generator.train()
    base_model.load_state_dict(torch.load("checkpoints/%s/pretrained" % directory_identifier_raw))
    # generator.load_state_dict(torch.load("checkpoints/%s/model" % directory_identifier_raw))

    dataloader = datautils.DataLoader(
        dataset, batch_size=batch_size // iter_per, shuffle=True, drop_last=False, pin_memory=True,
        num_workers=8
    )

    dataset.produce_keys("./data/%s/%s-keys-generated.txt" % (dataset_name, dataset_name))
    opt = AdamW(lr=2e-5, weight_decay=0.02,
                eps=1e-8, params=generator.parameters())

    if dataset_name in ["yelp_review", "news"]:
        generator.load_state_dict(torch.load("checkpoints/%s/warmup" % directory_identifier))

    if continue_training:
        generator.load_state_dict(torch.load("checkpoints/%s/model-finetune" % directory_identifier))
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
        selection_epoch = 2
        print(selection_epoch)
        generator.load_state_dict(torch.load("checkpoints/%s/model-finetune-%d-lambda-%s" % (directory_identifier, selection_epoch, args.bellman_reg)))
        generator.eval()
        if type(tokenizer) is NaiveTokenizer:
            tokenizer.close_vocab()
        fout = open("./data/%s/%s-generated-finetune.txt" % (dataset_name, dataset_name), "w")
        with open("./data/%s/%s-keys.txt" % (dataset_name, dataset_name), "r") as fin:
            for line in tqdm.tqdm(fin.readlines()):
                input_ids = torch.tensor([tokenizer.bos_token_id] + tokenizer.encode(line.strip()) + [tokenizer.sep_token_id])
                generated = generator.generate(input_ids=input_ids.cuda().unsqueeze(dim=0), max_length=300,
                                              num_beams=20, pad_token_id=tokenizer.eos_token_id)
                generated_str = tokenizer.decode(generated[0][len(input_ids):-1])
                print(generated_str, file=fout)
        fout.close()
        exit()
    bsz = args.batch_size // iter_per


    if dataset_name in {"common_gen"}:
        try:
            virtual_dataset = pickle.load(
                open("./data/%s/%s-sampled" % (dataset_name, directory_identifier_raw), "rb"))
            # raise NotImplementedError()
        except:
            dataset.produce_keys("./data/%s/%s-keys-generated.txt" % (dataset_name, dataset_name))
            fin = open("./data/%s/%s-keys-generated.txt" % (dataset_name, dataset_name), "r")
            virtual_dataset = LexicalCheckingDataset(tokenizer, expansion_num=48)
            virtual_dataset.add(fin, generator)
            fin.close()

            pickle.dump(virtual_dataset, open("./data/%s/%s-sampled" % (dataset_name, directory_identifier_raw), "wb"))
            exit()
        virtual_dataloader = datautils.DataLoader(
            virtual_dataset, batch_size=batch_size // iter_per, shuffle=True, drop_last=False, pin_memory=True,
            num_workers=8
        )
        stren_reg = args.bellman_reg
        for epoch_idx in range(10):
            iterator = tqdm.tqdm(virtual_dataloader)
            NLL_REDUCED = []
            DISCREPANCY_REDUCED = []
            for generated, mask, label, length in iterator:
                max_len = length.max().item() + 1
                if max_len > 300:

                    continue
                generated = generated.cuda()[:, 0:max_len]
                mask = mask.cuda()[:, 0:max_len-1]
                label = label.cuda()
                neufact_output = generator(generated, labels=torch.ones_like(generated).to(torch.float) * label.reshape(-1, 1))
                logits = neufact_output.loss
                reg_loss = neufact_output.reg_loss

                nll_reduced = -(logits * mask).sum(dim=-1)
                reg_loss_reduced = (reg_loss[:, :-1] * mask[:, 1:]).sum(dim=-1)
                NLL_REDUCED.append(nll_reduced.mean(dim=0).cpu().item())
                DISCREPANCY_REDUCED.append(reg_loss_reduced.mean(dim=0).cpu().item())
                if iter_count % iter_per == 0:
                    opt.zero_grad()
                loss = nll_reduced + float(stren_reg) * reg_loss_reduced
                (loss / iter_per).mean(dim=0).backward()
                if iter_count % iter_per == iter_per - 1:
                    opt.step()
                    if (iter_count // iter_per) % 10 == 0:
                        iterator.write("Iteration %d-%d, Loss %f, Reg Loss %f" % (
                        epoch_idx, iter_count // iter_per, NLL_REDUCED[-1], DISCREPANCY_REDUCED[-1]))
                # embed(); exit()
                iter_count += 1
            print("Now avg. nll_red=", np.mean(NLL_REDUCED))
            print("Now avg. reg_red=", np.mean(DISCREPANCY_REDUCED))
            epoch_idx += 1
            torch.save(generator.state_dict(), "checkpoints/%s/model-finetune-%d-lambda-%s" % (directory_identifier, epoch_idx, stren_reg))
            torch.save(generator.state_dict(),
                       "checkpoints/%s/model-finetune" % (directory_identifier))
            torch.save(opt.state_dict(), "checkpoints/%s/opt" % directory_identifier)
            torch.save((epoch_idx, iter_count), "checkpoints/%s/EpochIdx" % directory_identifier)
    elif dataset_name in {"yelp_review", "news"}:
        try:
            virtual_dataset = pickle.load(
                open("./data/%s/%s-sampled" % (dataset_name, directory_identifier_raw), "rb"))
            # raise NotImplementedError()
        except:
            # from torch.multiprocessing import Pool, Process, set_start_method
            # try:
            #     set_start_method('spawn')
            # except RuntimeError:
            #     pass
            dataset.produce_keys("./data/%s/%s-keys-generated.txt" % (dataset_name, dataset_name))
            with open("./data/%s/%s-keys-generated.txt" % (dataset_name, dataset_name), "r") as fin:
                virtual_dataset = ExactFormLexicalCheckingDataset(tokenizer, expansion_num=32)
                virtual_dataset.add(fin, generator)

            pickle.dump(virtual_dataset, open("./data/%s/%s-sampled" % (dataset_name, directory_identifier_raw), "wb"))
            exit()
        virtual_dataloader = datautils.DataLoader(
            virtual_dataset, batch_size=batch_size // iter_per, shuffle=True, drop_last=False, pin_memory=True,
            num_workers=8
        )

if __name__ == "__main__":
    main()