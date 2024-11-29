python main.py \
--cfg configs/cifar100.yaml \
--opts FED.IID True \
FED.NUM_CLIENTS 10 \
MISC.SIMILARITY ema_probs \
MODEL.ADAPTATION ours \
FED.BATCH_SIZE 200 \
FED.TEMPORAL_H 0.2 \
FED.SPATIAL_H 0.2 \
MISC.ADAPT_ALL True \
FED.AGG_FREQ 1 \
OPTIM.LR 0.002 \
MISC.RNG_SEED  42 \
MISC.EMA_PROBS_TEMP  1.0 \
MISC.TEMP 0.01 \
OPTIM.MT 0.9 \
OPTIM.RST 0.01 \
MISC.USE_IMLOSS  False \
MISC.USE_AUG False \
MISC.TEACHER_AVG True

