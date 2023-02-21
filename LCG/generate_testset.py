import datasets
import tqdm
# a_dev = datasets.load_dataset("common_gen", split="validation")
a_dev = datasets.load_dataset("common_gen", split="test")
test_base = dict()
for instance in tqdm.tqdm(a_dev):
    keyword_seq = " ".join(instance["concepts"])
    ref_seq = instance["target"]
    if keyword_seq in test_base:
        test_base[keyword_seq].append(ref_seq)
    else:
        test_base[keyword_seq] = [ref_seq]

with open("./data/common_gen/common_gen-keys.txt", "w") as fkeys:
    with open("./data/common_gen/common_gen-test.txt", "w") as ftest:
        for key in tqdm.tqdm(test_base):
            print(key, file=fkeys)
            for ref in test_base[key]:
                print(ref, file=ftest)
            print(file=ftest)
