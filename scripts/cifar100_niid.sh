python main.py --cfg configs/cifar100.yaml \
--opts FED.IID False \
MISC.SIMILARITY weights \
MODEL.ADAPTATION ours \
FED.BATCH_SIZE 10 \
FED.TEMPORAL_H 0.02 \
FED.SPATIAL_H 0.2 \
MISC.ADAPT_ALL False \
FED.AGG_FREQ 1