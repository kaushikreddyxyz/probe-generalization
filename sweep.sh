# python3 conditional_steer.py --only_learn 50337 --dataset "datasets/locations/" lora --lora_r 64 --layers 8
# python3 conditional_steer.py --only_learn 50337 --dataset "datasets/locations/" lora --lora_r 64 --layers 14
# python3 conditional_steer.py --only_learn 50337 --dataset "datasets/locations/" lora --lora_r 64 --layers 20
# python3 conditional_steer.py --only_learn 50337 --dataset "datasets/locations/" lora --lora_r 64 --layers 26


# for layer in $(seq 25 5 35); do
#     python3 conditional_steer.py --dataset "datasets/locations/" --lr 1.0 steer --layer $layer --hook_name mlp
# done


python3 conditional_steer.py --dataset "datasets/locations/" --lr 1.0 steer --layer 2 --hook_name mlp


python3 conditional_steer.py --lr 2e-3 --dataset "datasets/locations/" lora --lora_r 64 --only_learn 50337 --layers 4