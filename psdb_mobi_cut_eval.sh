basepath=$(cd `dirname $0`; pwd)
./evaluate_cut.py \
    --rec-path ${basepath}/data/psdb/val.rec \
	--network mobilenet_cut \
    --num-class 4 \
	--data-shape 300 \
	--cpu \
	--epoch 240 \
	--batch-size 1 \
    --prefix ${basepath}/output/psdbMobileNet_cut/ssd \
    --class-names 'pedestrian, head, head-shouler, upper-body' #  &> log_eval_psdb_mobi &

#	--gpus 0 \
