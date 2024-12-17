python main.py \
--cfg configs/cifar100.yaml \
--opts FED.IID True \
FED.NUM_CLIENTS 10 \
MISC.SIMILARITY feat \
MODEL.ADAPTATION ours \
FED.BATCH_SIZE 100 \
FED.TEMPORAL_H 0.1 \
FED.SPATIAL_H 0.2 \
MISC.ADAPT_ALL True \
FED.AGG_FREQ 1 \
OPTIM.LR 0.001 \
MISC.RNG_SEED  426 \
MISC.EMA_PROBS_TEMP  1.0 \
MISC.TEMP 0.01 \
OPTIM.MT 0.99 \
OPTIM.RST 0.00 \
MISC.USE_IMLOSS  False \
MISC.USE_AUG False \
MISC.TEACHER_AVG False

