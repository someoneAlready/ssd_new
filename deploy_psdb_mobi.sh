basepath=$(cd `dirname $0`; pwd)
./deploy.py \
	--network mobilenet \
    --num-class 4 \
    --prefix ${basepath}/output/psdbMobileNet/ssd \
	--data-shape 300 \
	--epoch 240


