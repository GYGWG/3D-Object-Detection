#!/bin/bash

# generate static model
python tools/export.py \
    --config configs/pointpillars/pointpillars_zw_pedestrian.yml \
    --model outputs/pointpillars_zw_pedestrian_1102/best_model/model.pdparams \
    --save_dir outputs/pointpillars_zw_pedestrian_1102/best_model
