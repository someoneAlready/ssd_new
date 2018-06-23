basepath=$(cd `dirname $0`; pwd)
./train_mobi.py \
    --train-path ${basepath}/data/psdb/train.rec \
    --num-class 4 \
    --prefix ${basepath}/output/psdbMobileNetLess/ssd \
    --val-path ${basepath}/data/psdb/val.rec \
	--label-width 1200 \
	--network mobilenetLess \
	--pretrained ${basepath}/model/mobilenet-ssd-512 \
	--finetune 1 \
	--data-shape 300 \
    --class-names 'pedestrian, head, head-shouler, upper-body' 

