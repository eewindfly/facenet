cd src
python infer.py \
  --file_ext png \
  --pretrained_model ~/models/facenet/20170131-234652/model-20170131-234652.ckpt-250000 \
  --data_dir ~/datasets/zackhsiao/FaceScrub/test_cropped/facescrub_aligned/ \
  --feature_dir ~/datasets/zackhsiao/FaceScrub/features/InceptionResnetV1_CenterLoss_MS-Celeb-1M/ \
  --image_size 160 \
  --model_def models.inception_resnet_v1 \
  --random_crop \
  --log_histograms \
  --gpu_memory_fraction 0.5
