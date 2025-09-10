from scipy.spatial.distance import pdist, squareform
import numpy as np
import os
import json
import pickle
import argparse
import pandas as pd

def load_jsonl(file_path):
    dat = open(file_path, 'r').readlines()
    dat = [json.loads(i) for i in dat]
    return dat

import numpy as np

def normalize_rows_l2(X):
    """
    对NumPy数组按行进行L2归一化，使得每行的L2范数等于1
    
    参数:
    X: numpy.array, 输入数组
    
    返回:
    numpy.array: 归一化后的数组，每行的L2范数为1
    """
    # 计算每行的L2范数
    row_norms = np.linalg.norm(X, axis=1, keepdims=True)
    
    # 避免除以零，将零范数的行设置为1（这样除以1不会改变值）
    row_norms[row_norms == 0] = 1
    
    # 归一化：每行除以其L2范数
    normalized_X = X / row_norms
    
    return normalized_X



def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_text_path', type=str, default='', required=True)
    parser.add_argument('--response_text_path', type=str, default='', required=True)
    parser.add_argument('--query_repre_path', type=str, default='', required=True)
    parser.add_argument('--response_repre_path', type=str, default='', required=True)
    parser.add_argument('--output_path', type=str, default='', required=True)
    parser.add_argument('--top_n', type=int, default=3, required=False)


    args = parser.parse_args()
    
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    query_files = load_jsonl(args.query_text_path)
    response_files = load_jsonl(args.response_text_path)
    response_ls = [i['text'] for i in response_files]

    query_represents = load_pkl(args.query_repre_path)
    response_represents = load_pkl(args.response_repre_path)

    top_n_retrieval_results = {}
    top_n_retrieval_results['source'] = [i['text'] for i in query_files]

    for domain in query_represents.keys():
        print(f"Processing domain: {domain}")
        top_n_retrieval_results[domain] = []
        query_matrix = normalize_rows_l2(query_represents[domain])
        response_matrix = normalize_rows_l2(response_represents[domain])
        cosine_sim_matrix = np.dot(query_matrix, response_matrix.T)
        top_n_indices = np.argsort(-cosine_sim_matrix, axis=1)[:, :args.top_n]

        if domain == '':
            domain = 'general'

        for i in range(args.top_n):
            col_name = domain + f'_top_{str(i+1)}'

            top_n_retrieval_results[col_name] = [response_ls[idx] for idx in top_n_indices[:, i]]
    top_n_retrieval_results = {k:v for k,v in top_n_retrieval_results.items() if len(v) > 0}
    df = pd.DataFrame(top_n_retrieval_results)
    df.to_csv(os.path.join(output_path, 'top_n_retrieval_results.csv'), index=False)

