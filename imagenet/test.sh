python imagenet.py --dataroot /gdata/ImageNet2012/ \
    --gpus 0 \
    -j 4 \
    --model mobilenetv2 \
    -b 900 \
    -e \
    --resume ../models/mobilenetv2_base/checkpoint/model_best.pth.tar