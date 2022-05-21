rm -rf /project/train/src_repo/yolov5/data/tempdata
mkdir /project/train/src_repo/yolov5/data/tempdata
rm -rf /project/train/src_repo/yolov5/runs/train/exp*
rm -rf /project/train/models/best.pt

cd /home/data 
ls -la

cp /home/data/831/*.xml /project/train/src_repo/yolov5/data/tempdata
cp /home/data/831/*.jpg  /project/train/src_repo/yolov5/data/images

python /project/train/src_repo/yolov5/initdata.py
python /project/train/src_repo/yolov5/train.py --weights /project/train/src_repo/yolov5/yolov5n.pt --data /project/train/src_repo/yolov5/setopt.yaml --batch 32 --epoch 100 
cp /project/train/src_repo/yolov5/runs/train/exp/weights/best.pt /project/train/models/best.pt
