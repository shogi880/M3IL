CUDA_VISIBLE_DEVICES=$2 OMP_NUM_THREADS=4 python3 main_il.py --random_seed=$1 \
--experiment_name=petnet_image \
--embedding_type=image_embedding \
--lambda_support=0.0 \
--lambda_query=0.1 \
--lambda_embedding=0 \
--lr=0.0005 \
--datasetdir='../../datasets' \
--eval=False \
--eval_trainenv=False \
--no_mujoco=False 