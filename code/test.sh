export CUDA_VISIBLE_DEVICES=0

BASE_WORK_DIR="results/gtav2cityscapes/"

# source only
python eval.py --config_file config/gtav2cityscapes/source_only.yaml --resume_from ${BASE_WORK_DIR}/source_only/best_iter.pth --work_dir ${BASE_WORK_DIR}/source_only/test

# warmup_at (adversarial training)
python eval.py --config_file config/gtav2cityscapes/warmup_at.yaml --resume_from ${BASE_WORK_DIR}/warmup_at/best_iter.pth --work_dir ${BASE_WORK_DIR}/warmup_at/test

# self-training 1
python eval.py --config_file config/gtav2cityscapes/sl_1.yaml --resume_from ${BASE_WORK_DIR}/sl/sl_1/best_iter.pth --work_dir ${BASE_WORK_DIR}/sl/sl_1/test

# self-training 2
python eval.py --config_file config/gtav2cityscapes/sl_2.yaml --resume_from ${BASE_WORK_DIR}/sl/sl_2/best_iter.pth --work_dir ${BASE_WORK_DIR}/sl/sl_2/test

# self-training 3
python eval.py --config_file config/gtav2cityscapes/sl_3.yaml --resume_from ${BASE_WORK_DIR}/sl/sl_3/best_iter.pth --work_dir ${BASE_WORK_DIR}/sl/sl_3/test
