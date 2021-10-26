CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 python3 main_il.py --random_seed=1 \
--experiment_name=mill_random_instruction --task_id=False \
--batch_size=64 --lr=0.0005 --loss_type=reparam_loss \
--lambda_support=0.0 --lambda_query=0.1 --lambda_embedding=$1 \
--datasetdir='../../datasets' --logdir='./log_1' \
--eval=False --eval_trainenv=False --no_mujoco=False \
--embedding_type=both_embedding  --sentence_type=both --sentence=15types \