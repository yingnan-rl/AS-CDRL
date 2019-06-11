#!bin/bash
for i in 1 5 10;
do
    python DDPG.py --env_name HalfCheetah-v2 --policy_name DDPG --seed $i --start_timesteps 10000 --exp_name HalfCheetah-v2-TPS-CI2-3000-20 --save_net 3000 --save_num 20
    
done
