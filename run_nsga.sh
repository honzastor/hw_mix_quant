python3 -W ignore src/run_nsga.py                           \
 --pretrained                                               \
 --pretrained_model mnv1_cifar10_100E_model_best.pth.tar    \
 --arch mobilenetv1                                         \
 --act_function relu                                        \
 --data src/pytorch/data/datasets/cifar10                   \
 --cache_directory nsga_experiments_caches                  \
 --cache_name eyeriss_mobilenetv1_cifar10_cache             \
 --dataset_name cifar10                                     \
 --workers 4                                                \
 --train_batch 256                                          \
 --test_batch 512                                           \
 --manual_seed 42                                           \
 --qat_epochs 3                                             \
 --parent_size 8                                            \
 --offspring_size 8                                         \
 --generations 3                                            \
 --lr 0.1                                                   \
 --wd 1e-5                                                  \
 --timeloop_architecture eyeriss                            \
 --timeloop_heuristic random                                \
 --total_valid 100                                          \
 --primary_metric memsize_words                             \
 --manual_seed 42                                           \
 --logs_dir logs_nsga_qat                                   \
 

