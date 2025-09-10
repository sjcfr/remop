
DATA_NAME='fiqa'

python ./src/remop/retrieve.py \
  --query_text_path ./data/$DATA_NAME/queries.jsonl \
  --response_text_path ./data/$DATA_NAME/corpus.jsonl \
  --query_repre_path ./output/embeddings/${DATA_NAME}/query/embeddings.pkl \
  --response_repre_path ./output/embeddings/${DATA_NAME}/response/embeddings.pkl \
  --output_path ./output/retrieval/${DATA_NAME}/
