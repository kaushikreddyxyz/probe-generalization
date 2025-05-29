
# python3 conditional_steer.py --only_learn 50337 --dataset "datasets/locations/" lora --lora_r 64 --layers 8
# python3 conditional_steer.py --only_learn 50337 --dataset "datasets/locations/" lora --lora_r 64 --layers 14
# python3 conditional_steer.py --only_learn 50337 --dataset "datasets/locations/" lora --lora_r 64 --layers 20
# python3 conditional_steer.py --only_learn 50337 --dataset "datasets/locations/" lora --lora_r 64 --layers 26

python3 conditional_steer.py --dataset "datasets/locations/" steer --layer 2 --hook_name mlp
python3 conditional_steer.py --dataset "datasets/locations/" steer --layer 2 --hook_name resid

./sweep.sh >> output.txt