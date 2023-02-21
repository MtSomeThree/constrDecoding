import datasets
import tqdm
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()


def nltk_pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_sentence(sentence):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    wordnet_tagged = map(lambda x: (x[0], nltk_pos_tagger(x[1])), nltk_tagged)
    lemmatized_sentence = []

    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)

N = 0
M = 0
with open("./data/common_gen/common_gen-generated-baseline.jsonl", "w") as fout:
    with open("./data/common_gen/common_gen-keys.txt", "r") as fkeys:
        with open("./data/common_gen/common_gen-generated-baseline.txt", "r") as ftest: #
            A, B = fkeys.readlines(), ftest.readlines()
            if len(A) == len(B):
                for (line_key, line_seq) in tqdm.tqdm(zip(A, B)):
                    keys = line_key.strip().split()
                    seq = lemmatize_sentence(line_seq.strip())
                    print("{\"concept_set\": \"%s\", \"pred_scene\": [\"%s\"]}" % ("#".join(keys), line_seq.strip()), file=fout)
                    # M += 1
                    M += len(keys)
                    included_keys = [key for key in keys if seq.count(key) > 0]
                    if len(keys) == len(included_keys):
                        # N += 1
                        N += len(included_keys)
                    else:
                        N += len(included_keys)
                        # print("Actual keys:", keys)
                        # print("Included keys:", included_keys)
                        # print("Generated samples:", line_seq)
            else:
                i = 0
                j = 0
                for i in range(len(A)):
                    cand_N = 0
                    keys = A[i].strip().split()
                    M += 1
                    while B[j].strip() != "":
                        seq = lemmatize_sentence(B[j].strip())
                        included_keys = [key for key in keys if seq.count(key) > 0]
                        cand_N = max(cand_N, len(included_keys))
                        j += 1
                    if len(keys) == cand_N:
                        N += 1
                    j += 1

print("baseline stats:")
print(N)
print(M)
print(N / M)

N = 0
M = 0
with open("./data/common_gen/common_gen-generated-finetune.jsonl", "w") as fout:
    with open("./data/common_gen/common_gen-keys.txt", "r") as fkeys:
        with open("./data/common_gen/common_gen-generated-finetune.txt", "r") as ftest: #
            A, B = fkeys.readlines(), ftest.readlines()
            if len(A) == len(B):
                for (line_key, line_seq) in tqdm.tqdm(zip(A, B)):
                    keys = line_key.strip().split()
                    seq = lemmatize_sentence(line_seq.strip())
                    print("{\"concept_set\": \"%s\", \"pred_scene\": [\"%s\"]}" % ("#".join(keys), line_seq.strip()), file=fout)
                    # M += 1
                    M += len(keys)
                    included_keys = [key for key in keys if seq.count(key) > 0]
                    if len(keys) == len(included_keys):
                        # N += 1
                        N += len(included_keys)
                    else:
                        N += len(included_keys)
                        # print("Actual keys:", keys)
                        # print("Included keys:", included_keys)
                        # print("Generated samples:", line_seq)
            else:
                i = 0
                j = 0
                for i in range(len(A)):
                    cand_N = 0
                    keys = A[i].strip().split()
                    M += 1
                    while B[j].strip() != "":
                        seq = lemmatize_sentence(B[j].strip())
                        included_keys = [key for key in keys if seq.count(key) > 0]
                        cand_N = max(cand_N, len(included_keys))
                        j += 1
                    if len(keys) == cand_N:
                        N += 1
                    j += 1


print("finetuned stats:")
print(N)
print(M)
print(N / M)