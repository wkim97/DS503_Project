cd taming-transformers && python main.py --base configs/VQGAN_blue.yaml -t True --gpus 0,1,2
cd ../KoDALLE && python train.py
