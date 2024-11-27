python main.py \
--cfg configs/cifar100.yaml \
--opts FED.IID True \
MISC.SIMILARITY weights \
MODEL.ADAPTATION ours_grad \
FED.BATCH_SIZE 10 \
FED.TEMPORAL_H 0.02 \
FED.SPATIAL_H 0.2 \
MISC.ADAPT_ALL True \
FED.AGG_FREQ 1 \
OPTIM.LR 0.01