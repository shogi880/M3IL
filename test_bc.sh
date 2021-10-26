CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python3 main_il.py --random_seed=$1 \
--experiment_name=task_id_1 --embedding_type=language_embedding  --sentence_type=language --sentence=sentence \
--batch_size=64 --lr=0.0005 --loss_type=reparam_loss --task_id=True \
--lambda_embedding=0 --lambda_support=0.0 --lambda_query=0.1 \
--datasetdir='../../datasets' --logdir='./experiment1' \
--eval=True --eval_trainenv=True --no_mujoco=False \
--eval_only=True --load=True --checkpoint_dir="./log_1" --checkpoint_iter=-1