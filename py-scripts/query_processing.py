import pickle
import re
import pandas as pd

def load_dictionaries():
    stop_words = pickle.load(open('data/stop_words.pickle', 'rb'))
    syn_csv = pd.read_csv('data/synonyms.csv')
    synonyms = {}
    for i in range(len(syn_csv)):
        synonyms[syn_csv['lemma'][i]] = set(re.split(';|\|', str(syn_csv['synonyms'][i]).lower()))
        synonyms[syn_csv['lemma'][i]].add(str(syn_csv['lemma'][i]).lower())

    return [stop_words, synonyms]

def load_queries():
    file_in = open('data/2019_200/queries.tsv', 'r')
    queries = {}
    for line in file_in.readlines():
        [id, text] = line.split('\t')
        queries[id] = re.findall(r"[\w']+", text.lower())
    file_in.close()
    return queries

def strip_queries(queries, dics):
    [stop_words, _] = dics

    for id in queries:
        queries[id] = [w for w in queries[id] if w not in stop_words]

def expand_queries(queries, dics):
    [_, synonyms] = dics

    for id in queries:
        fin = []
        for w in queries[id]:
            if w not in synonyms:
                fin.append(w)
                continue
            for t in synonyms[w]:
                fin.append(t)
        queries[id] = fin

def save_queries(queries):
    file_out = open('data/2019_200/queries_improved.tsv', 'w')

    for id in queries:
        file_out.write('{:}\t{:}\n'.format(id, ' '.join(queries[id])))

    file_out.close()

if __name__ == '__main__':
    dics = load_dictionaries()
    queries = load_queries()
    strip_queries(queries, dics)
    # expand_queries(queries, dics)
    # strip_queries(queries, dics)
    save_queries(queries)