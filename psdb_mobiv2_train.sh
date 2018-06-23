basepath=$(cd `dirname $0`; pwd)
./train_v2.py \
    --train-path ${basepath}/data/psdb/train.rec \
    --num-class 4 \
    --prefix ${basepath}/output/psdbMobileNet_v2/ssd \
    --val-path ${basepath}/data/psdb/val.rec \
	--label-width 1200 \
	--network mobilenet_v2 \
	--pretrained ${basepath}/mobilev2 \
	--lr 0.00003 \
	--finetune 5 \
	--data-shape 300 \
    --class-names 'pedestrian, head, head-shouler, upper-body' 

