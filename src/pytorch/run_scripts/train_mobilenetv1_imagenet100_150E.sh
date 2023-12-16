export CUDA_VISIBLE_DEVICES=0
python3 -W ignore ../train.py                       \
 --arch mobilenetv1                                 \
 --checkpoint_path ../checkpoints/mnv1_init         \
 --data ../data/datasets/imagenet100                \
 --dataset_name imagenet100                         \
 --workers 4                                        \
 --train_batch 128                                  \
 --test_batch 512                                   \
 --manual_seed 42                                   \
 --epochs 150                                       \
 --act_function relu                                \
 --lr 0.05                                          \
 --wd 0.00004                                       \
 --wandb                                            \
 --wandb_project MobileNet-QAT-Playground           \
 --wandb_entity honzastor                           \
 -v               