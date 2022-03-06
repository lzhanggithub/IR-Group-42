import pandas as pd
import csv
import numpy as np
import re
import pickle


def load_rel(rel, queryids, docids, file):
    for line in np.array(file):
        [q, _, d, r] = line
        q = int(q)
        d = int(d)
        r = int(r)
        queryids.add(q)
        docids.add(d)
        if q not in rel:
            rel[q] = {}
        rel[q][d] = r


def load_queries(queryids, queries, reader):
    i = 0
    for [id, text] in reader:
        i += 1
        print('\r{:6.2f}%'.format(i / len(reader)), end='')
        id = int(id)
        if id in queryids:
            queries[id] = text


def calculate_data():
    docids = set()
    queryids = set()

    train_rel = {}
    test_rel = {}
    valid_rel = {}

    load_rel(train_rel, queryids, docids, pd.read_csv('./data/L2R/train.tsv', delimiter='\t', header=None))
    load_rel(test_rel, queryids, docids, pd.read_csv('./data/L2R/test.txt', delimiter=' ', header=None))
    load_rel(valid_rel, queryids, docids, pd.read_csv('./data/L2R/valid.tsv', delimiter='\t', header=None))

    docs = {}
    doc_reader = csv.reader(open('./data/docs.tsv', 'r'), delimiter='\t')
    print('Collecting document text')
    i = 0
    for [id, text] in doc_reader:
        i += 1
        print('\r{:6.2f}%'.format(i / 88418.23), end='')
        id = int(id)
        if id in docids:
            docs[id] = text
    print('Done')

    queries = {}
    load_queries(queryids, queries, csv.reader(open('./data/queries.train.tsv', 'r'), delimiter='\t'))
    load_queries(queryids, queries, csv.reader(open('./data/queries.dev.tsv', 'r'), delimiter='\t'))
    load_queries(queryids, queries, csv.reader(open('./data/queries.eval.tsv', 'r'), delimiter='\t'))

    with open('./data/L2R/processed/train_rel.pickle', 'wb') as file:
        pickle.dump(train_rel, file)
    with open('./data/L2R/processed/test_rel.pickle', 'wb') as file:
        pickle.dump(test_rel, file)
    with open('./data/L2R/processed/valid_rel.pickle', 'wb') as file:
        pickle.dump(valid_rel, file)
    with open('./data/L2R/processed/docs.pickle', 'wb') as file:
        pickle.dump(docs, file)
    with open('./data/L2R/processed/queries.pickle', 'wb') as file:
        pickle.dump(queries, file)


def load_data():
    [train_rel, test_rel, valid_rel, documents, queries, stop_words] = [None] * 6

    with open('./data/L2R/processed/train_rel.pickle', 'rb') as file:
        train_rel = pickle.load(file)
    with open('./data/L2R/processed/test_rel.pickle', 'rb') as file:
        test_rel = pickle.load(file)
    with open('./data/L2R/processed/valid_rel.pickle', 'rb') as file:
        valid_rel = pickle.load(file)
    with open('./data/L2R/processed/docs.pickle', 'rb') as file:
        documents = pickle.load(file)
    with open('./data/L2R/processed/queries.pickle', 'rb') as file:
        queries = pickle.load(file)
    with open('data/stop_words.pickle', 'rb') as file:
        stop_words = pickle.load(file)

    return [train_rel, test_rel, valid_rel, documents, queries, stop_words]


def index_terms(documents, stop_words):
    terms = {}
    i = 0
    count = len(documents)
    for doc_id in documents:
        i += 1
        print('\r{:6.2f}% '.format(i / count * 100), end='')
        terms[doc_id] = {}
        for term in re.findall(r"[\w']+", documents[doc_id].lower()):
            if term in stop_words:
                continue
            if term not in terms[doc_id]:
                terms[doc_id][term] = 0
            terms[doc_id][term] += 1
    return terms


def calculate_index(data):
    [train_rel, test_rel, valid_rel, documents, queries, stop_words] = data
    indexed_ques = index_terms(queries, stop_words)
    print('done')
    with open('data/L2R/processed/indexed_ques.pickle', 'wb') as file:
        pickle.dump(indexed_ques, file)

    indexed_docs = index_terms(documents, stop_words)
    print('done')
    with open('./data/L2R/processed/indexed_docs.pickle', 'wb') as file:
        pickle.dump(indexed_docs, file)

    with open('./data/L2R/processed/indexed_docs.pickle', 'rb') as file:
        indexed_docs = pickle.load(file)

    inverted_index = {}
    i = 0
    for did in indexed_docs:
        i += 1
        print('\r{:6.2f}% '.format(i / len(indexed_docs) * 100), end='')
        for term in indexed_docs[did]:
            if term not in inverted_index:
                inverted_index[term] = set()
            inverted_index[term].add(did)
    with open('./data/L2R/processed/inverted_index.pickle', 'wb') as file:
        pickle.dump(inverted_index, file)


def load_index():
    [indexed_ques, indexed_docs, inverted_index] = [None] * 3

    with open('data/L2R/processed/indexed_ques.pickle', 'rb') as file:
        indexed_ques = pickle.load(file)

    with open('./data/L2R/processed/indexed_docs.pickle', 'rb') as file:
        indexed_docs = pickle.load(file)

    with open('./data/L2R/processed/inverted_index.pickle', 'rb') as file:
        inverted_index = pickle.load(file)

    return [indexed_ques, indexed_docs, inverted_index]


def calc_tf(index, qid, did):
    [indexed_ques, indexed_docs, inverted_index] = index

    count = 0
    summ = 0
    for term in indexed_ques[qid]:
        count += indexed_ques[qid][term]
        if term in indexed_docs[did]:
            summ += indexed_ques[qid][term] * indexed_docs[did][term] / np.sum(
                [indexed_docs[did][tid] for tid in indexed_docs[did]])

    if count == 0:
        return 0
    return summ / count


def calc_idf(index, did, idf_cache):
    [indexed_ques, indexed_docs, inverted_index] = index

    count = 0
    summ = 0
    for term in indexed_docs[did]:
        if term not in idf_cache:
            idf_cache[term] = np.log2(len(indexed_docs) / len(inverted_index[term]))
        count += indexed_docs[did][term]
        summ += indexed_docs[did][term] * idf_cache[term]

    if count == 0:
        return 0
    return summ / count


# def calc_bm25(n, tf, doc_len, avg_len, qf, N):
def calc_bm25(data, index, f_vals, qid, did, avg_len):
    [train_rel, test_rel, valid_rel, documents, queries, stop_words] = data
    [indexed_ques, indexed_docs, inverted_index] = index
    [tf_vals, idf_vals] = f_vals

    r = 0
    R = 0

    # constants
    k1 = 1.2
    k2 = 100
    b = 0.75

    doc_len = np.sum([indexed_docs[did][t] for t in indexed_docs[did]])

    K = k1 * ((1 - b) + b * (doc_len / avg_len))

    score = 0
    for term in indexed_ques[qid]:
        if term not in indexed_docs[did]:
            continue

        tf = indexed_docs[did][term]
        qf = indexed_ques[qid][term]
        n = len(inverted_index[term])

        sub1 = R - r + 0.5
        upper = (r + 0.5) / sub1 * (k1 + 1) * tf * (k2 + 1) * qf
        sub2 = len(documents) - n - R + r + 0.5
        lower = (n - r + 0.5) / sub2 * (K + tf) * (k2 + qf)
        score += max(-1, np.log10(upper / lower))

    return score


def process_tf(data, index):
    [train_rel, test_rel, valid_rel, documents, queries, stop_words] = data
    [indexed_ques, indexed_docs, inverted_index] = index

    print('tf_values train set:')
    tf_vals = {}
    i = 0
    for query in train_rel:
        i += 1
        print('\r{:6.2f}% '.format(i / len(train_rel) * 100), end='')
        tf_vals[query] = {}
        for document in train_rel[query]:
            tf_vals[query][document] = calc_tf(index, query, document)
    print('done')

    print('tf_values test set:')
    i = 0
    for query in test_rel:
        i += 1
        print('\r{:6.2f}% '.format(i / len(test_rel) * 100), end='')
        if query not in tf_vals:
            tf_vals[query] = {}
        for document in test_rel[query]:
            if document in tf_vals[query]:
                print('\nwell this is embarrasing')
                continue
            tf_vals[query][document] = calc_tf(index, query, document)
    print('done')

    print('tf_values valid set:')
    i = 0
    for query in valid_rel:
        i += 1
        print('\r{:6.2f}% '.format(i / len(valid_rel) * 100), end='')
        if query not in tf_vals:
            tf_vals[query] = {}
        for document in valid_rel[query]:
            if document in tf_vals[query]:
                print('\nwell this is embarrasing')
                continue
            tf_vals[query][document] = calc_tf(index, query, document)
    print('done')

    with open('./data/L2R/processed/tf_values.pickle', 'wb') as file:
        pickle.dump(tf_vals, file)


def load_tf():
    with open('./data/L2R/processed/tf_values.pickle', 'rb') as file:
        return pickle.load(file)


def process_idf(data, index):
    [train_rel, test_rel, valid_rel, documents, queries, stop_words] = data
    [indexed_ques, indexed_docs, inverted_index] = index

    print('idf_values:')
    idf_vals = {}
    idf_cache = {}
    i = 0
    for document in documents:
        i += 1
        print('\r{:6.2f}% '.format(i / len(documents) * 100), end='')
        idf_vals[document] = calc_idf(index, document, idf_cache)
    print('done')

    with open('./data/L2R/processed/idf_values.pickle', 'wb') as file:
        pickle.dump(idf_vals, file)


def load_idf():
    with open('./data/L2R/processed/idf_values.pickle', 'rb') as file:
        return pickle.load(file)


def process_bm25(data, index, f_vals):
    [train_rel, test_rel, valid_rel, documents, queries, stop_words] = data
    [indexed_ques, indexed_docs, inverted_index] = index
    [tf_vals, idf_vals] = f_vals

    avg_len = np.average([np.sum([indexed_docs[d][t] for t in indexed_docs[d]]) for d in indexed_docs])

    print('bm25_values train set:')
    bm25_vals = {}
    i = 0
    for query in train_rel:
        i += 1
        print('\r{:6.2f}% '.format(i / len(train_rel) * 100), end='')
        bm25_vals[query] = {}
        for document in train_rel[query]:
            bm25_vals[query][document] = calc_bm25(data, index, f_vals, query, document, avg_len)
    print('done')

    print('bm25_values test set:')
    i = 0
    for query in test_rel:
        i += 1
        print('\r{:6.2f}% '.format(i / len(test_rel) * 100), end='')
        if query not in bm25_vals:
            bm25_vals[query] = {}
        for document in test_rel[query]:
            if document in bm25_vals[query]:
                print('\nwell this is embarrasing')
                continue
            bm25_vals[query][document] = calc_bm25(data, index, f_vals, query, document, avg_len)
    print('done')

    print('bm25_values valid set:')
    i = 0
    for query in valid_rel:
        i += 1
        print('\r{:6.2f}% '.format(i / len(valid_rel) * 100), end='')
        if query not in bm25_vals:
            bm25_vals[query] = {}
        for document in valid_rel[query]:
            if document in bm25_vals[query]:
                print('\nwell this is embarrasing')
                continue
            bm25_vals[query][document] = calc_bm25(data, index, f_vals, query, document, avg_len)
    print('done')

    with open('./data/L2R/processed/bm25_values.pickle', 'wb') as file:
        pickle.dump(bm25_vals, file)


def load_bm25():
    with open('./data/L2R/processed/bm25_values.pickle', 'rb') as file:
        return pickle.load(file)

def export_features(data, index, f_vals, bm25_vals):
    [train_rel, test_rel, valid_rel, documents, queries, stop_words] = data
    [indexed_ques, indexed_docs, inverted_index] = index
    [tf_vals, idf_vals] = f_vals

    file_out = open('data/L2R/feature_dev_train_plus.txt', 'w')

    i = 0
    for query in train_rel:
        i += 1
        print('\r{:6.2f}% '.format(i / len(train_rel) * 100), end='')
        for document in train_rel[query]:
            line = '{:} qid:{:} 1:{:} 2:{:} 3:{:} 4:{:} 5:{:} #docid = {:}\n'.format(
                train_rel[query][document],
                query,
                len(indexed_docs[document]),
                np.sum([indexed_docs[document][term] for term in indexed_docs[document]]),
                tf_vals[query][document],
                idf_vals[document],
                bm25_vals[query][document],
                document)
            file_out.write(line)

    file_out.close()

    file_out = open('data/L2R/feature_dev_test_plus.txt', 'w')

    i = 0
    for query in test_rel:
        i += 1
        print('\r{:6.2f}% '.format(i / len(test_rel) * 100), end='')
        for document in test_rel[query]:
            line = '{:} qid:{:} 1:{:} 2:{:} 3:{:} 4:{:} 5:{:} #docid = {:}\n'.format(
                test_rel[query][document],
                query,
                len(indexed_docs[document]),
                np.sum([indexed_docs[document][term] for term in indexed_docs[document]]),
                tf_vals[query][document],
                idf_vals[document],
                bm25_vals[query][document],
                document)
            file_out.write(line)
    file_out.close()

    file_out = open('data/L2R/feature_dev_valid_plus.txt', 'w')
    i = 0
    for query in valid_rel:
        i += 1
        print('\r{:6.2f}% '.format(i / len(valid_rel) * 100), end='')
        for document in valid_rel[query]:
            line = '{:} qid:{:} 1:{:} 2:{:} 3:{:} 4:{:} 5:{:} #docid = {:}\n'.format(
                valid_rel[query][document],
                query,
                len(indexed_docs[document]),
                np.sum([indexed_docs[document][term] for term in indexed_docs[document]]),
                tf_vals[query][document],
                idf_vals[document],
                bm25_vals[query][document],
                document)
            file_out.write(line)
    file_out.close()

if __name__ == '__main__':
    calculate_data()
    data = load_data()
    calculate_index(data)
    index = load_index()
    process_tf(data, index)
    process_idf(data, index)
    f_vals = [load_tf(), load_idf()]
    process_bm25(data, index, f_vals)
    bm25_vals = load_bm25()
    export_features(data, index, f_vals, bm25_vals)
