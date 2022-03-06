import json
import pandas as pd
import csv
import numpy as np

def process():
    docids = set()
    queryids = set()
    run = {}
    for [q, d, _] in np.array(pd.read_csv('./data/2019_200/run.tsv', delimiter='\t', header=None)):
        q = int(q)
        d = int(d)
        if q not in run:
            run[q] = []
        run[q].append(d)
        docids.add(d)
        queryids.add(q)

    relevance = {}
    for [q, _, d, r] in np.array(pd.read_csv('./data/2019_200/qrels.tsv', delimiter=' ', header=None)):
        q = int(q)
        d = int(d)
        r = int(r)
        queryids.add(q)
        docids.add(d)
        if q not in relevance:
            relevance[q] = {}
        relevance[q][d] = r

    docs = {}
    doc_reader = csv.reader(open('./data/docs.tsv', 'r'), delimiter='\t')
    for [id, text] in doc_reader:
        id = int(id)
        if id in docids:
            docs[id] = text

    queries = {}
    quer_reader = csv.reader(open('./data/2019_200/queries.tsv', 'r'), delimiter='\t')
    for [id, text] in quer_reader:
        id = int(id)
        if id in queryids:
            queries[id] = text

    # with open('./data/2019_200/processed/run.json', 'w') as file:
    #     json.dump(run, file)
    # with open('./data/2019_200/processed/relevance.json', 'w') as file:
    #     json.dump(relevance, file)
    # with open('./data/2019_200/processed/docs.json', 'w') as file:
    #     json.dump(docs, file)
    # with open('./data/2019_200/processed/queries.json', 'w') as file:
    #     json.dump(queries, file)

def load():
    [run, rel, doc, que] = [None] * 4
    with open('./data/2019_200/processed/run.json', 'r') as file:
        run = json.load(file)
    with open('./data/2019_200/processed/relevance.json', 'r') as file:
        rel = json.load(file)
    with open('./data/2019_200/processed/docs.json', 'r') as file:
        doc = json.load(file)
    with open('./data/2019_200/processed/queries.json', 'r') as file:
        que = json.load(file)

    return [run, rel, doc, que]


# Behavior on unretrieved relevant documents [Why weren’t these relevant documents retrieved within the top 1000?]
def process_rel_not_ret(run, rel, doc, que):
    for qid in run.keys():
        if qid not in rel:
            continue
        print('QUERY:{:7} {:}'.format(qid,que[qid]))
        for did in rel[qid].keys():
            if did not in rel[qid]:
                continue
            if rel[qid][did] <= 0:
                continue
            if did not in run[qid]:
                print('\tDOC:{:7} {:}'.format(did, doc[did]))

def process_not_rel_but_ret(run, rel, doc, que):
    for qid in run:
        if qid not in rel:
            continue
        print('QUERY:{:7} {:}'.format(qid,que[qid]))
        for pos, did in enumerate(run[qid]):
            if did in rel[qid] and rel[qid][did] > 0:
                continue
            if did not in doc:
                continue
            print('\tpos:{:4} DOC:{:7} {:}'.format(pos+1, did, doc[did]))




if __name__=='__main__':
    # process()
    [run, rel, doc, que] = load()
    # process_rel_not_ret(run, rel, doc, que)
    process_not_rel_but_ret(run, rel, doc, que)