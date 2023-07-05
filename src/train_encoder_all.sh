# Linear regression
CUDA_VISIBLE_DEVICES=0 python train.py --conf conf/encoder/linear_regression.yaml &

# Classification
CUDA_VISIBLE_DEVICES=1 python train.py --conf conf/encoder/linear_classification.yaml &

# Ridge
CUDA_VISIBLE_DEVICES=2 python train.py --conf conf/encoder/ridge_1.0.yaml &
CUDA_VISIBLE_DEVICES=3 python train.py --conf conf/encoder/ridge_0.5.yaml &
CUDA_VISIBLE_DEVICES=4 python train.py --conf conf/encoder/ridge_0.25.yaml &
CUDA_VISIBLE_DEVICES=5 python train.py --conf conf/encoder/ridge_0.1.yaml &

# MTL
CUDA_VISIBLE_DEVICES=6 python train.py --conf conf/encoder/linear_and_logistic.yaml &
CUDA_VISIBLE_DEVICES=7 python train.py --conf conf/encoder/ridge_mtl.yaml