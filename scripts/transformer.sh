PROBLEM=languagemodel_ptb10k
MODEL=transformer
HPARAMS=transformer_base # it was transformer_base_single_gpu on 1-GPU test
SCRIPT_PATH=/home/nishome/ydwu/app/ehorovod/exps/tensor2tensor/scripts

DATA_DIR=$SCRIPT_PATH/t2t_data
TMP_DIR=$SCRIPT_PATH/t2t_datagen
TRAIN_DIR=$SCRIPT_PATH/t2t_train/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

## Generate data
#t2t-datagen \
#  --data_dir=$DATA_DIR \
#  --tmp_dir=$TMP_DIR \
#  --problem=$PROBLEM
#
#mv $TMP_DIR/tokens.vb $DATA_DIR

## multi-server scripts

#mpirun -np 1 --host node109:1 -x TF_CONFIG='{"cluster": {"ps": ["10.113.216.109:10014"], "chief": ["10.113.216.109:10013"], "worker": ["10.113.216.109:10012"]}, "task": {"type": "ps", "index": 0}, "environment": "cloud"}' \
#  -x CUDA_VISIBLE_DEVICES= \
#   t2t-trainer --schedule=run_std_server >/dev/null 2>&1 &
#
#
#mpirun -np 1 --host node109:1 -x TF_CONFIG='{"cluster": {"ps": ["10.113.216.109:10014"], "chief": ["10.113.216.109:10013"], "worker": ["10.113.216.109:10012"]}, "task": {"type": "worker", "index": 0}, "environment": "cloud"}'\
#  -x CUDA_VISIBLE_DEVICES=4,5,6,7 \
#  t2t-trainer --data_dir=$DATA_DIR --problem=$PROBLEM --model=$MODEL \
#  --hparams_set=$HPARAMS --output_dir=$TRAIN_DIR --hparams='batch_size=124' \
#  --master=grpc://10.113.216.109:10012 --ps_replicas=1 --worker_replicas=2 --worker_gpu=4 --worker_id=1 --ps_gpu=0 --schedule=train --worker_job='/job:worker' >/dev/null 2>&1 &
#
#time mpirun -np 1 --host node109:1 -x TF_CONFIG='{"cluster": {"ps": ["10.113.216.109:10014"], "chief": ["10.113.216.109:10013"], "worker": ["10.113.216.109:10012"]}, "task": {"type": "chief", "index": 0}, "environment": "cloud"}'\
#  -x CUDA_VISIBLE_DEVICES=0,1,2,3 \
#  t2t-trainer --data_dir=$DATA_DIR --problem=$PROBLEM --model=$MODEL \
#  --hparams_set=$HPARAMS --output_dir=$TRAIN_DIR --hparams='batch_size=124'  \
#  --master=grpc://10.113.216.109:10013 --ps_replicas=1 --worker_replicas=2 --worker_gpu=1 --worker_id=0 --ps_gpu=0  --schedule=train --worker_job='/job:chief' 

## Single server scripts
#mpirun -np 1 --host node118:1 -x INIT_CLUSTER_SIZE=4 -x HOROVOD_LOG_LEVEL=TRACE \
#  -x MY_DEVICE=4 \
#  t2t-trainer --data_dir=$DATA_DIR --problem=$PROBLEM --model=$MODEL \
#  --hparams_set=$HPARAMS --output_dir=$TRAIN_DIR --hparams='batch_size=2048' --worker_gpu=1  --use_edl=True &
#
#mpirun -np 1 --host node118:1 -x INIT_CLUSTER_SIZE=4 -x HOROVOD_LOG_LEVEL=TRACE \
#  -x MY_DEVICE=5 \
#  t2t-trainer --data_dir=$DATA_DIR --problem=$PROBLEM --model=$MODEL \
#  --hparams_set=$HPARAMS --output_dir=$TRAIN_DIR --hparams='batch_size=2048' --worker_gpu=1  --use_edl=True &

mpirun -np 1 --host node118:1 -x INIT_CLUSTER_SIZE=2 -x HOROVOD_LOG_LEVEL=TRACE \
  -x MY_DEVICE=6 \
  t2t-trainer --data_dir=$DATA_DIR --problem=$PROBLEM --model=$MODEL \
  --hparams_set=$HPARAMS --output_dir=$TRAIN_DIR --hparams='batch_size=2048' --worker_gpu=1  --use_edl=True &

time mpirun -np 1 --host node118:1 -x INIT_CLUSTER_SIZE=2 -x HOROVOD_LOG_LEVEL=TRACE \
  -x MY_DEVICE=7 \
  t2t-trainer --data_dir=$DATA_DIR --problem=$PROBLEM --model=$MODEL \
  --hparams_set=$HPARAMS --output_dir=$TRAIN_DIR --hparams='batch_size=2048' --worker_gpu=1  --use_edl=True 
