docker rm -f facenet
NV_GPU=0,1 nvidia-docker run -it -p 8888:8888 -p 6006:6006 --name facenet \
  -v $(pwd):/root/workspace/facenet \
  -v /media/shared_data/VGG_FACE_DATASET/VGG_FACE_DATASET:/root/workspace/datasets/vggface \
  eewindfly/facenet
