#!/bin/sh
#OAR -n test_script
#OAR -t exotic
#OAR -l /gpu=2,walltime= 22:22:22
#OAR --stdout result_finetuning.txt
python finetuning_metaeval.py
