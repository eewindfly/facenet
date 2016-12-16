docker rm -f facenet
NV_GPU=0 nvidia-docker run -it -p 8888:8888 -p 6006:6006 --name facenet \
  -v $(pwd):/root/workspace/facenet \
  -v /media/shared_data:/root/datasets \
  -v ~/logs:/root/logs \
  -v ~/models:/root/models \
  eewindfly/facenet
