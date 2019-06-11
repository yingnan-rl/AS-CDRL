#!bin/bash
for i in 5 10;
do
    python DDPG.py --env_name Walker2d-v2 --policy_name DDPG --seed $i --start_timesteps 10000 --exp_name Walker2d-v2-TPS-CI2-5000-30 --save_net 5000 --save_num 30
    
done
