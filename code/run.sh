export CUDA_VISIBLE_DEVICES=0

BASE_WORK_DIR="results/gtav2cityscapes/"

# source only
python main.py --config_file config/gtav2cityscapes/source_only.yaml --work_dir ${BASE_WORK_DIR}/source_only

# warmup (adversarial training)
python main.py --config_file config/gtav2cityscapes/warmup_at.yaml --resume_from ${BASE_WORK_DIR}/source_only/best_iter.pth --work_dir ${BASE_WORK_DIR}/warmup_at

# self-training round 1
python main.py --config_file config/gtav2cityscapes/sl_1.yaml --resume_from ${BASE_WORK_DIR}/warmup_at/last_iter.pth --pseudo_resume_from ${BASE_WORK_DIR}/warmup_at/best_iter.pth --work_dir ${BASE_WORK_DIR}/sl/sl_1

# create pseudo-label in round 1 with the same size at round 2
mv ${BASE_WORK_DIR}/sl/sl_1/pseudo/ ${BASE_WORK_DIR}/sl/sl_1/pseudo_1280/
python main.py --config_file config/gtav2cityscapes/sl_1_1536.yaml --resume_from ${BASE_WORK_DIR}/warmup_at/last_iter.pth --pseudo_resume_from ${BASE_WORK_DIR}/warmup_at/best_iter.pth --work_dir ${BASE_WORK_DIR}/sl/sl_1

# self-training round 2
python main.py --config_file config/gtav2cityscapes/sl_2.yaml --resume_from ${BASE_WORK_DIR}/sl/sl_1/last_iter.pth --pseudo_resume_from ${BASE_WORK_DIR}/sl/sl_1/epoch_5.pth --work_dir ${BASE_WORK_DIR}/sl/sl_2

# # # self-training round 3
python main.py --config_file config/gtav2cityscapes/sl_3.yaml --resume_from ${BASE_WORK_DIR}/sl/sl_2/last_iter.pth --pseudo_resume_from ${BASE_WORK_DIR}/sl/sl_2/epoch_5.pth --work_dir ${BASE_WORK_DIR}/sl/sl_3
