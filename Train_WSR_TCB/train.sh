#!/bin/bash
LOG=log/MKEI_Word-`date +%Y-%m-%d-%H-%M-%S`.log
MPLBACKEND=Agg python ./train.py --gpu 2 2>&1 | tee $LOG