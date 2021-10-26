CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python3 main_il.py --random_seed=$1 \
--experiment_name=mill_init_image_task_id_1 --init_image=True --task_id=True \
--embedding_type=both_embedding  --sentence_type=$2 --sentence=True \
--batch_size=64 --lr=0.0005 --loss_type=reparam_loss \
--lambda_embedding=1e-7 --lambda_support=0.0 --lambda_query=0.1 \
--datasetdir='../../datasets' --logdir='./result' \
--eval=True --eval_trainenv=True --no_mujoco=False \
--eval_only=True --load=True --checkpoint_dir="./log_model" --checkpoint_iter=-1
--reparam_embedding_type=None --fusion_type=add