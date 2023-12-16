python3 -W ignore src/run_nsga.py                           \
 --pretrained                                               \
 --pretrained_model mnv1_cifar10_100E_model_best.pth.tar    \
 --arch mobilenetv1                                         \
 --act_function relu                                        \
 --data src/pytorch/data/datasets/cifar10                   \
 --dataset_name cifar10                                     \
 --workers 4                                                \
 --train_batch 256                                          \
 --test_batch 512                                           \
 --manual_seed 42                                           \
 --qat_epochs 5                                             \
 --parent_size 10                                           \
 --offspring_size 10                                        \
 --generations 5                                            \
 --lr 0.1                                                   \
 --wd 1e-5                                                  \
 --timeloop_architecture eyeriss                            \
 --timeloop_heuristic random                                \
 --total_valid 100                                          \
 --primary_metric edp                                       \
 --manual_seed 42                                           \
 

