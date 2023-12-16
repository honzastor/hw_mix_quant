# Author: Jan Klhufek (iklhufek@fit.vut.cz)

import os
import shutil
from typing import List
from tqdm import tqdm

# NOTE: subset taken as inspiration from haq repository: https://github.com/mit-han-lab/haq/blob/master/lib/utils/imagenet100.txt
IMAGENET100_LABELS = ['n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475', 'n01496331',
                      'n01498041', 'n01514668', 'n01514859', 'n01518878', 'n01530575', 'n01531178',
                      'n01532829', 'n01534433', 'n01537544', 'n01558993', 'n01560419', 'n01580077',
                      'n01582220', 'n01592084', 'n01601694', 'n01608432', 'n01614925', 'n01616318',
                      'n01622779', 'n01629819', 'n01630670', 'n01631663', 'n01632458', 'n01632777',
                      'n01641577', 'n01644373', 'n01644900', 'n01664065', 'n01665541', 'n01667114',
                      'n01667778', 'n01669191', 'n01675722', 'n01677366', 'n01682714', 'n01685808',
                      'n01687978', 'n01688243', 'n01689811', 'n01692333', 'n01693334', 'n01694178',
                      'n01695060', 'n01697457', 'n01698640', 'n01704323', 'n01728572', 'n01728920',
                      'n01729322', 'n01729977', 'n01734418', 'n01735189', 'n01737021', 'n01739381',
                      'n01740131', 'n01742172', 'n01744401', 'n01748264', 'n01749939', 'n01751748',
                      'n01753488', 'n01755581', 'n01756291', 'n01768244', 'n01770081', 'n01770393',
                      'n01773157', 'n01773549', 'n01773797', 'n01774384', 'n01774750', 'n01775062',
                      'n01776313', 'n01784675', 'n01795545', 'n01796340', 'n01797886', 'n01798484',
                      'n01806143', 'n01806567', 'n01807496', 'n01817953', 'n01818515', 'n01819313',
                      'n01820546', 'n01824575', 'n01828970', 'n01829413', 'n01833805', 'n01843065',
                      'n01843383', 'n01847000', 'n01855032', 'n01855672']


def create_imagenet_subset(dataset_path: str, subset_path: str, labels: List[str]) -> None:
    """
    Create a subset of the ImageNet dataset containing only specified labels.

    Args:
        dataset_path (str): Path to the original ImageNet dataset.
        subset_path (str): Path where the subset of ImageNet will be created.
        labels (List[str]): List of label names to be included in the subset.
    """
    assert os.path.exists(dataset_path), f"'{dataset_path}'" + ' path for dataset not found!'
    traindir = os.path.join(dataset_path, 'train')
    valdir = os.path.join(dataset_path, 'val')
    assert os.path.exists(traindir), f"'{traindir}'" + ' path for training data not found!'
    assert os.path.exists(valdir), f"'{valdir}'" + ' path for validation data not found!'

    # Create subset directory structure if it doesn't exist
    os.makedirs(os.path.join(subset_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(subset_path, 'val'), exist_ok=True)

    for label in tqdm(labels, desc=f"Creating ImageNet{len(labels)} subset"):
        # Paths for the full dataset's label data
        dataset_train_dir = os.path.join(traindir, label)
        dataset_val_dir = os.path.join(valdir, label)
        assert os.path.exists(dataset_train_dir), f"'{dataset_train_dir}' the label '{label}' does not exist in the training dataset diretory!"
        assert os.path.exists(dataset_val_dir), f"'{dataset_val_dir}' the label '{label}' does not exist in the validation dataset diretory!"

        # Paths for the subset dataset's label data
        subset_train_dir = os.path.join(subset_path, 'train', label)
        subset_val_dir = os.path.join(subset_path, 'val', label)

        # Copy directories
        shutil.copytree(dataset_train_dir, subset_train_dir)
        shutil.copytree(dataset_val_dir, subset_val_dir)


if __name__ == '__main__':
    create_imagenet_subset(dataset_path='../datasets/imagenet', subset_path='../datasets/imagenet100', labels=IMAGENET100_LABELS)
