keys=# INSERT YOUR API KEY(S) HERE

#evaporate code clse ie
python run_profiler.py \
    --data_lake fda_510ks \
    --num_attr_to_cascade 50 \
    --num_top_k_scripts 10 \
    --train_size 10 \
    --combiner_mode ws \
    --use_dynamic_backoff \
    --KEYS ${keys}\
    --data_dir  /data/fda_510ks/data/evaporate/fda-ai-pmas/510k \
    --base_data_dir /data/evaporate/data/fda_510ks \
    --gold_extractions_file /data/evaporate/data/fda_510ks/table.json \
#evaporate code open ie
python run_profiler.py \
    --data_lake fda_510ks \
    --num_attr_to_cascade 50 \
    --num_top_k_scripts 10 \
    --train_size 10 \
    --combiner_mode ws \
    --use_dynamic_backoff \
    --KEYS ${keys}\
    --do_end_to_end \
    --data_dir  /data/fda_510ks/data/evaporate/fda-ai-pmas/510k \
    --base_data_dir /data/evaporate/data/fda_510ks \
    --gold_extractions_file /data/evaporate/data/fda_510ks/table.json \
