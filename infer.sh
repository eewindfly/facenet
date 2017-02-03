cd src
python infer.py \
  --pretrained_model ~/models/facenet/20161215-070543 \
  --logs_base_dir ~/logs/facenet/ \
  --data_dir ~/datasets/zackhsiao/MegaFace/official_cropped/daniel/FlickrFinal2/ \
  --image_size 192 \
  --model_def models.inception_resnet_v1 \
  --weight_decay 2e-4 \
  --optimizer RMSPROP \
  --random_crop \
  --center_loss_factor 2e-4 \
  --log_histograms \
  --gpu_memory_fraction 0.5
