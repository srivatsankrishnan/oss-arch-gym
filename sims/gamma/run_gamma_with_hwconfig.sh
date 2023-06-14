cd ./src/GAMMA
python main.py --fitness1 latency --fitness2 power  --epochs 10 \
              --model vgg16 --singlelayer 1 --hwconfig hw_config.m
cd ../../






