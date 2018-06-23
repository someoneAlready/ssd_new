basepath=$(cd `dirname $0`; pwd)
./evaluate_cut.py \
    --rec-path ${basepath}/data/val.rec \
	--network mobilenet_cut \
	--data-shape 512 \
	--gpus 0 \
	--epoch 240 \
	--batch-size 1 \
    --prefix ${basepath}/output/pascalMobileNet_cut/ssd \


#	--cpu \
