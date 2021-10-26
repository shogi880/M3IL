CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python3 main_il.py --random_seed=1 \
--experiment_name=mill_init_image_random_instruction --init_image=True --task_id=False \
--batch_size=64 --lr=0.0005 --loss_type=reparam_loss \
--lambda_support=0.0 --lambda_query=0.1 --lambda_embedding=1e-7 \
--datasetdir='../../datasets' --logdir='./log_model' \
--eval=False --eval_trainenv=False --no_mujoco=False \
--embedding_type=both_embedding  --sentence_type=both --sentence=random \