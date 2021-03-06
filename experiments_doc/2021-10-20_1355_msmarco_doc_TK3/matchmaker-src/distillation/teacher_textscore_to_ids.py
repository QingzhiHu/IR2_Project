#
# generate id files from raw text scored pairwise teacher output files
# -------------------------------
#

import argparse
from collections import defaultdict
import os
import sys
from tqdm import tqdm
sys.path.append(os.getcwd())

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--collection', action='store', dest='collection',
                    help='the full collection file location', required=True)

parser.add_argument('--query', action='store', dest='query',
                    help='query.tsv', required=True)

parser.add_argument('--in-file', action='store', dest='in_file',
                    help='the query output file location', required=True)

parser.add_argument('--out-file', action='store', dest='out_file',
                    help='scores and ids', required=True)


args = parser.parse_args()


#
# load data 
# -------------------------------
# 
queries = {}
with open(args.query,"r",encoding="utf8") as query_file:
    for line in tqdm(query_file):
        ls = line.split("\t") # id<\t>text ....
        queries[ls[1].rstrip()] = ls[0]

docs = {}
with open(args.collection,"r",encoding="utf8") as collection_file:
    for line in tqdm(collection_file):
        ls = line.split("\t") # id<\t>text ....
        docs[ls[1].rstrip()] = ls[0]

#
# produce output
# -------------------------------
#  
stats = defaultdict(int)
with open(args.out_file,"w",encoding="utf8") as out_file:
    with open(args.in_file,"r",encoding="utf8") as in_file:

        for line in tqdm(in_file):
            line = line.split("\t") # scorpos scoreneg query docpos docneg

            try:
                q_id = queries[line[2]]
                doc_pos_id = docs[line[3]]
                doc_neg_id = docs[line[4].rstrip()]
    
                out_file.write(line[0]+"\t"+line[1]+"\t"+q_id+"\t"+doc_pos_id+"\t"+doc_neg_id+"\n")

            except KeyError as e:
                stats["key_error"]+=1


for key, val in stats.items():
    print(f"{key}\t{val}")