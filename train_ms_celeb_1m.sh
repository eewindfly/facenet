cd src
python facenet_train_classifier.py \
  --pretrained_model ~/models/facenet/20170131-234652/model-20170131-234652.ckpt-250000 \
  --logs_base_dir ~/logs/facenet/ \
  --models_base_dir ~/models/facenet/ \
  --data_dir ~/datasets/MsFace/MsCeleb_align_crop \
  --image_size 299 \
  --model_def models.inception_resnet_v1 \
  --lfw_dir ~/datasets/LFW/crop \
  --weight_decay 2e-4 \
  --optimizer RMSPROP \
  --learning_rate -1 \
  --max_nrof_epochs 80 \
  --keep_probability 0.8 \
  --random_crop \
  --random_flip \
  --learning_rate_schedule_file ../data/learning_rate_schedule_classifier_long.txt \
  --center_loss_factor 2e-4 \
  --lfw_file_ext jpg \
  --log_histograms \
  --gpu_memory_fraction 0.5
