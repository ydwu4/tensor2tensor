PROBLEM=languagemodel_ptb10k
MODEL=transformer
HPARAMS=transformer_base # it was transformer_base_single_gpu on 1-GPU test

DATA_DIR=~/Projects/tensor2tensor/t2t_data
TMP_DIR=~/Projects/tensor2tensor/t2t_datagen
TRAIN_DIR=~/Projects/tensor2tensor/t2t_train/$PROBLEM/$MODEL-$HPARAMS

#mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

## Generate data
#t2t-datagen \
#  --data_dir=$DATA_DIR \
#  --tmp_dir=$TMP_DIR \
#  --problem=$PROBLEM
 
mpirun -np 16 --host node124:8,node121:8 -x AUTOGRAPH_VERBOSITY=0  \
  --mca btl_tcp_if_include ib0 \
  /home/nishome/w00397525/miniconda3/envs/gcc4.8.5/bin/t2t-trainer --data_dir=$DATA_DIR --problem=$PROBLEM --model=$MODEL \
  --hparams_set=$HPARAMS --output_dir=$TRAIN_DIR --hparams='batch_size=2048' --worker_gpu=8 --eval_steps=1
