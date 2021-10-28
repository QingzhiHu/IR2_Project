import os
import pandas as pd
import random
import json
import argparse


def create_folder():
    cwd = os.getcwd()
    path = os.path.join(cwd, 'qualitative')
    os.makedirs(path, exist_ok=True)
    return path


def show_validations(args, qualitative_path):
    cwd = os.getcwd()
    experiments_path = os.path.join(cwd, args.path)
    experiments = os.listdir(experiments_path)  # returns list of experiments

    output_name = 'best-validation-output.txt'
    column_names = ["query_id", "doc_id", "rank", "output_value"]
    total_frame = pd.DataFrame([], columns=column_names)
    for experiment in experiments:
        experiment_path = os.path.join(experiments_path, experiment)
        if output_name in os.listdir(experiment_path):
            print(f"Processing: {experiment}")
            df = pd.read_csv(os.path.join(experiment_path, output_name), delimiter="\t", names=column_names)
            df = df[(df['query_id'] == args.query) & (df['doc_id'] == args.doc)]   # only keeps info for query and doc
            total_frame = pd.concat([total_frame, df])
    total_frame.to_csv(os.path.join(qualitative_path, "ranking-scores.tsv"), sep="\t", index=False)


def create_test_file(args, qualitative_path):
    folder_name = "training_data/validation"

    cwd = os.getcwd()
    validation_path = os.path.join(cwd, folder_name)
    test_file = os.listdir(validation_path)[0]
    chunksize = 10 ** 6

    column_names = ["query_id", "doc_id", "query", "doc"]
    total_frame = pd.DataFrame([], columns=column_names)
    for i, chunk in enumerate(pd.read_csv(os.path.join(validation_path, test_file), delimiter="\t", chunksize=chunksize, names=column_names)):
        query_frame = chunk[chunk['query_id'] == args.query]
        total_frame = pd.concat([total_frame, query_frame])

    total_frame.to_csv(os.path.join(qualitative_path, f"top1000-query{args.query}.dev"), sep="\t", index=False, header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='experiments')
    parser.add_argument('--query', type=int, default=2)
    parser.add_argument('--doc', type=int, default=4339068)
    args = parser.parse_args()

    path = create_folder()
    show_validations(args, path)
    create_test_file(args, path)

