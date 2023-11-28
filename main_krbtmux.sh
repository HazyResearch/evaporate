#!/bin/bash
# - snap: https://ilwiki.stanford.edu/doku.php?id=snap-servers:snap-servers and support il-action@cs.stanford.edu
# - live server stats: https://ilwiki.stanford.edu/doku.php?id=snap-servers:snap-gpu-servers-stats
#8 a100 80GB
ssh brando9@ampere1.stanford.edu
ssh brando9@skampere1.stanford.edu
#10 Quadro RTX 8000 48GB
ssh brando9@hyperturing1.stanford.edu
ssh brando9@hyperturing2.stanford.edu
#10 RTX A4000 16GB
ssh brando9@mercury1.stanford.edu
ssh brando9@mercury2.stanford.edu

# allows mouse scrolling
tput rmcup

source $AFS/.bashrc
conda activate maf
export CUDA_VISIBLE_DEVICES=5; export SLURM_JOBID=$(python -c "import random;print(random.randint(0, 1_000_000))"); echo $SLURM_JOBID;
echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES; echo SLURM_JOBID = $SLURM_JOBID; echo hostname = $(hostname)
ulimit -n 120000; ulimit -Sn; ulimit -Hn;
nvidia-smi;hostname
(echo "GPU_ID PID UID APP" ; for GPU in 0 1 2 3 ; do for PID in $( nvidia-smi -q --id=${GPU} --display=PIDS | awk '/Process ID/{print $NF}') ; do echo -n "${GPU} ${PID} " ; ps -up ${PID} | awk 'NR-1 {print $1,$NF}' ; done ; done) | column -t

export CUDA_VISIBLE_DEVICES=3,4,5,6; export SLURM_JOBID=$(python -c "import random;print(random.randint(0, 1_000_000))"); echo $SLURM_JOBID;
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7; export SLURM_JOBID=$(python -c "import random;print(random.randint(0, 1_000_000))"); echo $SLURM_JOBID;
echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES; echo SLURM_JOBID = $SLURM_JOBID; echo hostname = $(hostname)

python -c "import uutils; uutils.torch_uu.gpu_test()"
python -c "import torch; print(torch.cuda.get_device_capability());print('if >=8 you can use bfloat16');"
python -c "import torch; print(torch.bfloat16);"

# - start krbtmux
#pkill -9 tmux -u brando9; pkill -9 krbtmux -u brando9; pkill -9 reauth -u brando9; pkill -9 python -u brando9; pkill -9 wandb-service* -u brando9;
krbtmux
reauth

ssh brando9@ampere1.stanford.edu
ssh brando9@hyperturing1.stanford.edu
ssh brando9@hyperturing2.stanford.edu
tmux ls

tmux new -s 0
tmux new -s 1
tmux new -s 2
tmux new -s rand
tmux new -s rand0
tmux new -s rand1
tmux new -s rand2
tmux new -s rand3
tmux new -s rand4
tmux new -s rand5
tmux new -s rand6
tmux new -s rand7
tmux new -s rand8
tmux new -s rand9
tmux new -s rand10
tmux new -s rand11
tmux new -s rand12
tmux new -s rand13
tmux new -s rand14
tmux new -s rand15
tmux new -s rand16
tmux new -s rand17
tmux new -s rand18
tmux new -s rand19
tmux new -s rand20
tmux new -s rand21
tmux new -s rand22
tmux new -s rand23
tmux new -s rand24
reauth

# - Min setup code for ru
reauth

source $AFS/.bashrc.lfs
conda activate maf
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -k2 -nr | head -n 1 | awk -F ', ' '{print $1}')
echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES

# --- Run
# # -- Make data sets HF
# # conda activate maf
# # python ~/massive-autoformalization-maf/maf-src/data_utils/textbook2hf_dataset.py

# # -- Train
# #login to hf to use llama2, get your hf token eg cat ~/keys/brandos_hf_token.txt
# huggingface-cli login
# #login to hf to use llama2, get your hf token eg cat ~/keys/brandos_hf_token.txt
# python ~/massive-autoformalization-maf/maf-src/af_train/unpaired_pytorch_hf_training.py

# # - create accelerate config & run script using accelerate
# accelerate config
# # accelerate config -- ~/massive-autoformalization-maf/configs/accelerate/default_config_ddp_bm.yaml 
# #login to hf to use llama2, get your hf token eg cat ~/keys/brandos_hf_token.txt
# huggingface-cli login
# # get top 3 free gpus & export so every child process gets it
# export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t ',' -k2 -nr | head -n 3 | awk -F ', ' '{printf "%s,",$1}' | sed 's/,$//')
# echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES
# # accelerate launch --config_file ~/.cache/huggingface/accelerate/default_config.yaml ~/massive-autoformalization-maf/maf-src/af_train/unpaired_pytorch_hf_training.py
# accelerate launch --config_file ~/massive-autoformalization-maf/configs/accelerate/default_config_bm.yaml ~/massive-autoformalization-maf/maf-src/af_train/unpaired_pytorch_hf_training.py

# # -- Maf eval data extraction
# # conda activate evaporate
# conda activate maf

# -- Run data extraction
# num_attr_to_cascade --> number of attributes to extract functions for.
# do_end_to_end --> if True, means learn/infer schema/OpenIE.
# num_top_k_scripts --> number of functions to extract for each attribute.
# use_dynamic_backoff --> if set (store True) means use generated function to extract.

# train_size --> The --train_size parameter controls how many sample documents Evaporate will use during the training stage to generate the extraction functions.
# combiner_mode --> The combiner_mode parameter in Evaporate controls how the outputs from multiple extraction functions are aggregated to produce the final metadata for each document.

# export keys_brando=$(cat ~/data/openai_api_key.txt)
# export keys_koyejolab=$(cat ~/data/openai_api_key_koyejolab_brando.txt)
export keys=$(cat ~/data/openai_api_key_koyejolab_brando.txt)

# conda activate maf
conda activate evaporate

# see /lfs/skampere1/0/brando9/evaporate/evaporate/configs.py to see the name of data lake that is at the config e.g., /lfs/skampere1/0/brando9/evaporate/evaporate/configs.py
DATA_LAKE=TODO  # TODO, what does code expect?
# you need to run from: https://chat.openai.com/g/g-KV0CvoH8Y-python-excellent-comments-doc-strings-types/c/125c002a-3e77-41a9-8130-b3ae8da26d4c
cd ~/evaporate/

python ~/evaporate/evaporate/run_profiler_maf.py \
    --data_lake $DATA_LAKE \
    --do_end_to_end False \
    --num_attr_to_cascade 25 \
    --num_top_k_scripts 5 \
    --train_size 10 \
    --combiner_mode ws \
    --use_dynamic_backoff True \
    --KEYS ${keys}

# python ~/evaporate/evaporate/run_profiler.py \
#     --data_lake rudin_chapter1_md \
#     --do_end_to_end False \
#     --num_attr_to_cascade 25 \
#     --num_top_k_scripts 5 \
#     --train_size 10 \
#     --combiner_mode ws \
#     --use_dynamic_backoff True \
#     --KEYS ${keys}

# python ~/evaporate/evaporate/run_profiler.py \
#     --data_lake small_debug_lin_alg_textbook \
#     --num_attr_to_cascade 15 \
#     --num_top_k_scripts 5 \
#     --train_size 10 \
#     --combiner_mode mv \

# -- other option is to run `echo $SU_PASSWORD | /afs/cs/software/bin/reauth` inside of python, right?
export JOB_PID=$!
echo $OUT_FILE
echo $ERR_FILE
echo JOB_PID = $JOB_PID
echo SLURM_JOBID = $SLURM_JOBID
