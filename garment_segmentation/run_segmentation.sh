#!/bin/bash

for div_num in {200..400}
do
   bsub -n 2 -W 24:00  -R "rusage[mem=10000]" python3 -u segment_garment.py --folder_path /cluster/scratch/gboeshertz/data --div $div_num
done
