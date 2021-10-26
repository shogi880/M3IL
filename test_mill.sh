CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python3 main_il.py --random_seed=1 \
--experiment_name=mill_random_instruction_1 \
--embedding_type=both_embedding \
--sentence_type=$1 --sentence=15types --task_id=False \
--batch_size=64 --lr=0.0005 --loss_type=reparam_loss \
--lambda_embedding=$2 --lambda_support=0.0 --lambda_query=0.1 \
--datasetdir='../../datasets' --logdir='./ablation/2_hypa' \
--eval=True --eval_trainenv=True --no_mujoco=False \
--eval_only=True --load=True --checkpoint_dir="./log_1" --checkpoint_iter=-1 \