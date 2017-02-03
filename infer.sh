cd src
python infer.py \
  --pretrained_model ../../models/facenet/20161116-234200 \
  --logs_base_dir ~/logs/facenet/ \
  --data_dir ~/datasets/CASIA/crop \
  --image_size 160 \
  --model_def models.inception_resnet_v1 \
  --weight_decay 2e-4 \
  --optimizer RMSPROP \
  --random_crop \
  --center_loss_factor 2e-4 \
  --log_histograms \
  --gpu_memory_fraction 0.5
