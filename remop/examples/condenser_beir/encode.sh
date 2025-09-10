
DATA_NAME='fiqa'
domains='general,medical,legal,financial,educational,technical'

python ./src/remop/encode.py \
  --output_path ./output/embeddings/${DATA_NAME}/query/ \
  --model_path ./model \
  --input_path ./data/$DATA_NAME/queries.jsonl \
  --domains $domains

python src/remop/encode.py \
  --output_path output/embeddings/${DATA_NAME}/response/ \
  --model_path model \
  --input_path data/$DATA_NAME/corpus.jsonl \
  --domains $domains