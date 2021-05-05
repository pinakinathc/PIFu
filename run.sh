#!/bin/bash

cd /vol/research/sketchcaption/extras/original/PIFu

# Overfit Experiment on 20 garments
# rm val.txt

# /user/HS500/pc00725/miniconda3/envs/PIFu/bin/python -m apps.train_shape --name=ndf_20_overfit --dataroot=/vol/research/NOBACKUP/CVSSP/scratch_4weeks/pinakiR/extra/training_20_data/ --random_flip --random_scale --random_trans --random_multiview --num_views=3 --batch_size=1 --no_num_eval --no_gen_mesh --sigma=0.3 --num_threads=12


# Complete Experiment
cp val.txt.bkp val.txt

/user/HS500/pc00725/miniconda3/envs/PIFu/bin/python -m apps.train_shape --name=ndf_overall --dataroot=/vol/research/NOBACKUP/CVSSP/scratch_4weeks/pinakiR/extra/training_data/ --random_flip --random_scale --random_trans --random_multiview --num_views=3 --batch_size=1 --no_num_eval --no_gen_mesh --sigma=0.3 --num_threads=12	
