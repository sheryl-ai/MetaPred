#!/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -l walltime=4:00:00
#PBS -N session1_default
#PBS -A course
#PBS -q ShortQ

#cd $PBS_O_WORKDIR
# cnn configuration
python main.py --method='cnn' --metatrain_iterations=10000 --meta_batch_size=32 --update_batch_size=4 --meta_lr=0.0001 --update_lr=1e-5 --num_updates=4 --n_total_batches=500000
# rnn configuration
python main.py --method='rnn' --metatrain_iterations=10000 --meta_batch_size=32 --update_batch_size=4 --meta_lr=0.0001 --update_lr=1e-5 --num_updates=4 --n_total_batches=500000
echo All done
