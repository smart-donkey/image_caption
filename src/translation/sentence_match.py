#encoding=utf8
import codecs
import json

import numpy as np
from openpyxl import load_workbook

# py2
import c_engine_api
# py3
# from . sentence_mapping import c_engine_api

dim = 50
max_count = -1
engine = c_engine_api.ChineseNLPAPI('pku')
stopwords = []

word_vectors = {}


def load_word_vectors(file_name):
    # id_2_vectors = {}
    voc = set()
    with codecs.open(file_name, encoding='utf8',errors='ignore') as f:
        header = f.readline()
        count, dim = header.split()
        read_count = 0
        for line in f:
            # print line
            if  max_count > 0 and read_count >= max_count:
                break
            if read_count % 100000 == 0:
                print(read_count, ' completed.')
            try:
                word2vec = line.split()
                id = len(word_vectors)
                word_vectors[word2vec[0]] = np.asarray(word2vec[1:]).astype(float)
                voc.add(word2vec[0])
            except UnicodeError:
                print(line)
                print(len(word_vectors))
            else:
                read_count += 1

    return word_vectors, count, dim, voc


def pre_process_sentences(sentences, voc):
    post_processed_sentences = []
    for sentence in sentences:
        explan = sentence['explanation']
        wsl = []
        if len(explan) > 330:
            for sub in explan.split(u'。'):
                if len(sub) > 0:
                    wsl_sub, pos_sub = engine.tokenize(sub)
                    wsl += wsl_sub
        else:
            wsl, pos = engine.tokenize(explan)

        # what the hell you are doing
        not_hit_count = 0
        for ws in wsl:
            if ws not in voc:
                not_hit_count += 1
        if 1 - (not_hit_count + 0.) / len(wsl) > -1:
            post_processed_sentences.append(sentence)
            sentence['seg'] = wsl
    return post_processed_sentences


def read_stop_words():
    global stopwords
    with codecs.open("StopWords_cn.txt", encoding='utf8') as f:
        for line in f:
            stopwords.append(line.strip())

# Simply average all the word vectors
def sentence_2_tensor(tokens, word_vectors):
    sentence_vector = np.zeros((dim,))
    global stopwords
    for token in tokens:
        if stopwords:
            if token in stopwords:
                continue

        if token in word_vectors.keys():
            sentence_vector += word_vectors[token]
        else:
            sentence_vector += np.random.rand(dim) * 0.01
            # print token
    sentence_vector /= len(tokens) + 0.
    return sentence_vector

def sentence_2_vector(tokens, local_word_vectors):
    sentence_vector = np.zeros((dim,))
    global stopwords
    for token in tokens:
        if stopwords:
            if token in stopwords:
                continue

        if token in local_word_vectors.keys():
            sentence_vector += word_vectors[token]
        else:
            # random_encoding = np.random.rand(dim) * 0.01
            # sentence_vector += random_encoding
            # # global vec
            # word_vectors[token] = random_encoding
            raise "word vec not exists"
    sentence_vector /= len(tokens) + 0.
    return sentence_vector

def get_local_word2vector(local_voc):
    local_word_vectors = {}
    global word_vectors
    for v in local_voc:
        if v in word_vectors:
            local_word_vectors[v] = word_vectors[v]
        else:
            random_encoding = np.random.rand(dim) * 0.01
            # global vec
            word_vectors[v] = random_encoding
            local_word_vectors[v] = random_encoding

    return local_word_vectors

def sentences_2_matrix(sentences):
    global word_vectors
    target_sentences_vector = np.zeros((len(sentences), dim), dtype=float)
    sentences_tokens = []
    local_voc = set()
    for i in range(len(sentences)):
        # print i
        # tokens, _ = engine.tokenize(sentences[i]['explanation'])
        tokens = sentences[i]['seg']
        sentences_tokens.append(tokens)
        for token in tokens:
            local_voc.add(token)
        # sentence_vector = sentence_2_vector(tokens, word_vectors)
        # target_sentences_vector[i] = sentence_vector

    # get local w2v
    local_word_vectors = get_local_word2vector(local_voc)
    for i in range(len(sentences_tokens)):
        sentence_vector = sentence_2_vector(sentences_tokens[i], local_word_vectors)
        target_sentences_vector[i] = sentence_vector

    return target_sentences_vector


def sentences_2_tensor(sentences, word_vectors):
    target_sentences_vector = np.zeros((len(sentences), dim), dtype=float)
    sentences_tokens = []
    local_voc = set()
    for i in range(len(sentences)):
        # print i
        # tokens, _ = engine.tokenize(sentences[i]['explanation'])
        tokens = sentences[i]['seg']
        sentences_tokens.append(tokens)
        for token in tokens:
            local_voc.add(token)
        # sentence_vector = sentence_2_vector(tokens, word_vectors)
        # target_sentences_vector[i] = sentence_vector

    # get local w2v
    local_word_vectors = get_local_word2vector(local_voc, word_vectors)
    for i in range(len(sentences_tokens)):
        sentence_vector = sentence_2_vector(sentences_tokens[i], local_word_vectors)
        target_sentences_vector[i] = sentence_vector

    return target_sentences_vector

def closest_sentence(source_sentence, target_sentences):
    # dist_square = np.sum(target_sentences ** 2, axis=1, keepdims=True) + np.sum(source_sentence ** 2) - 2 * np.dot(source_sentence[np.newaxis,], target_sentences.T).T
    # dists = dist_square
    # arg_list = np.argsort(dists.ravel())
    dist_square = source_sentence - target_sentences
    arg_list = np.linalg.norm(dist_square, axis=1)
    return np.argsort(arg_list)


def get_chinese_poem(match_target_file):
    wb = load_workbook(match_target_file)
    ws = wb['data']
    data = []
    for row in ws.rows:
        if row[3].value is not None:
            data.append({'author': row[0], 'title': row[1].value, 'content': row[2].value, 'explanation': row[3].value})
    return  data


def get_quotes(file_name):
    with codecs.open(file_name, encoding='utf8', errors='ignore') as f:
        data = [{'content': l.strip(), 'explanation': l.strip()} for l in f if l.strip()]
    return data

if __name__ == "__main__":
    # global dim
    # dim = 50
    read_stop_words()
    target_sentences = get_chinese_poem("tangshi.xlsx")
    word2vec, _, _, voc = load_word_vectors("/home/roger/BaiduZhidao_wv%d.txt" % dim)
    target_sentences = pre_process_sentences(target_sentences, voc)
    target_sentence_matrix = sentences_2_matrix(target_sentences)

    print('start to match, sentences cout:', len(target_sentences))
    input_file = "../neuraltalk2/vis/vis_processed.json"
    output_file = "../neuraltalk2/vis/vis_processed_matched_300.json"
    cou = 0

    with codecs.open(input_file, encoding='utf8') as f:
        js = json.load(f)
        for entity in (js):
            print("entity %d" % cou)
            cou += 1
            caption = entity["caption_zh"]
            source_sentence, _  = engine.tokenize(caption)
            print(" ".join(source_sentence))
            local_voc = set()
            for token in source_sentence:
                local_voc.add(token)
            local_word_2_vec = get_local_word2vector(local_voc)
            source_sentence_vector = sentence_2_vector(source_sentence, local_word_2_vec)
            arg_list = closest_sentence(source_sentence_vector, target_sentence_matrix)

            entity["caption_zh_matched_org_1"] = target_sentences[arg_list[0]]['content']
            entity["caption_zh_matched_org_2"] = target_sentences[arg_list[1]]['content']
            entity["caption_zh_matched_org_3"] = target_sentences[arg_list[2]]['content']
            entity["caption_zh_matched_org_last_1"] = target_sentences[arg_list[-1]]['content']
            entity["caption_zh_matched_org_last_2"] = target_sentences[arg_list[-2]]['content']
            entity["caption_zh_matched_org_last_3"] = target_sentences[arg_list[-3]]['content']
            # entity["caption_zh_matched_exp"] = target_sentences[id]['explanation']

        with codecs.open(output_file, "w") as f_o:
            f_o.write(json.dumps(js))
