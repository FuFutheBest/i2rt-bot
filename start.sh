 bash third_party/i2rt/scripts/reset_all_can.sh 
 export SI_I2RT_CAN_LEADER_LEFT_SERIAL="$(udevadm info --query=property --path=/sys/class/net/can1 | sed -n 's/^ID_SERIAL=//p')"
 export SI_I2RT_CAN_FOLLOWER_LEFT_SERIAL="$(udevadm info --query=property --path=/sys/class/net/can3 | sed -n 's/^ID_SERIAL=//p')"
 export SI_I2RT_CAN_LEADER_RIGHT_SERIAL="$(udevadm info --query=property --path=/sys/class/net/can0 | sed -n 's/^ID_SERIAL=//p')"
 export SI_I2RT_CAN_FOLLOWER_RIGHT_SERIAL="$(udevadm info --query=property --path=/sys/class/net/can2 | sed -n 's/^ID_SERIAL=//p')"
 export SI_I2RT_YAM_GRIPPER_TYPE=crank_4310
 source .venv/bin/activate
 python -m incubator.nedf3.I2RT_DM3.web   --host 127.0.0.1   --port 8080

