export CUDA_VISIBLE_DEVICES=0
python3 -W ignore ../../eval.py                                                 \
 --pretrained                                                                   \
 --arch mobilenetv1                                                             \
 --pretrained_model ../checkpoints/mnv1_QAT_sym_pertensor/chkpt_mobilenetv1_imagenet100_20231215_074621/model_after_qat.pth.tar          \
 --data ../data/datasets/imagenet100                                            \
 --dataset_name imagenet100                                                     \
 --workers 4                                                                    \
 --test_batch 512                                                               \
 --manual_seed 42                                                               \
 --qat                                                                          \
 --symmetric_quant                                                              \
 --quant_setting uniform                                                        \
 --uniform_width 6                                                              \
 -v                                                                             \
