CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=4 python3 main_il.py --random_seed=$2 \
--experiment_name=petnet_image --embedding_type=image_embedding --sentence_type=both --sentence=sentence \
--lambda_support=0.0 --lambda_query=0.1 --lambda_embedding=0 \
--batch_size=64 --lr=0.0005 --loss_type=reparam_loss \
--datasetdir='../../datasets' --logdir='./log_1' \
--eval=False --eval_trainenv=False --no_mujoco=False \
--image_type=first