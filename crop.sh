cd src
source env.sh

cd align
python align_dataset_mtcnn.py ~/workspace/datasets/vggface/data/ ~/workspace/datasets/vggface/crop_mtcnnpy_182 --image_size 182 --margin 44
