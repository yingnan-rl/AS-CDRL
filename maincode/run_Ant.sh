#!bin/bash
for i in 1 5 10;
do
    python DDPG.py --env_name Ant-v2 --policy_name DDPG --seed $i --start_timesteps 10000 --exp_name Ant-v2-TPS-CI2-6000-20-sample-1 --save_net 6000 --save_num 20 --sample_times 1
    
done
