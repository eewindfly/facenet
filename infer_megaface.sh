cd src
python infer.py \
  --file_ext jpg \
  --pretrained_model ~/models/facenet/20170131-005910/model-20170131-005910.ckpt-80000 \
  --data_dir ~/datasets/zackhsiao/MegaFace/data/daniel/FlickrFinal2/ \
  --feature_dir ~/datasets/zackhsiao/MegaFace/features/InceptionResnetV1_CenterLoss_CASIA/ \
  --image_size 299 \
  --model_def models.inception_resnet_v1 \
  --random_crop \
  --log_histograms \
  --gpu_memory_fraction 0.5
