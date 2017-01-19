cd src
python facenet_train_classifier.py \
  --logs_base_dir ~/logs/facenet/ \
  --models_base_dir ~/models/facenet/ \
  --data_dir ~/datasets/CASIA/crop \
  --image_size 160 \
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
  --log_histograms
