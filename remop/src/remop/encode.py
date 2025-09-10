import os
from FlagEmbedding import FlagModel
from scipy.spatial.distance import pdist, squareform
import json
import argparse
import numpy as np
import pickle
import os

def load_jsonl(file_path):
    dat = open(file_path, 'r').readlines()
    dat = [json.loads(i) for i in dat]
    return dat

def save_pkl(vectors, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(vectors, f)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='', required=True)
    parser.add_argument('--output_path', type=str, default='', required=True)
    parser.add_argument('--model_path', type=str, default='./model')
    parser.add_argument('--domains', type=str, default=None)

    args = parser.parse_args()
    
    input_path = args.input_path
    output_path = args.output_path
    model_path = args.model_path

    os.makedirs(output_path, exist_ok=True)

    dat_ls = load_jsonl(input_path)
    repre_dict = {}
    if args.domains is None:
        model = FlagModel(model_path, 
                        query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                        use_fp16=True)
        text_ls = [i['text'] for i in dat_ls]
        repre_vectors = model.encode( text_ls )
        repre_dict['general'] = repre_vectors
    else:
        domains = args.domains.split(',')
        domains.append('')
        for domain in domains:
            print(f"Processing domain: {domain}")

            model = FlagModel(model_path, 
                            query_instruction_for_retrieval=f"为这个{domain}句子生成表示以用于检索相关文章：",
                            use_fp16=True)
            text_ls = [i['text'] for i in dat_ls]
            repre_vectors = model.encode( text_ls )
            repre_dict[domain] = repre_vectors
    save_pkl(repre_dict, os.path.join(output_path, 'embeddings.pkl'))



