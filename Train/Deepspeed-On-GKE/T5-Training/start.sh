cp /etc/ssh/ssh_config .
chmod 666 ssh_config
echo "StrictHostKeyChecking no" > ssh_config
sudo cp ssh_config /etc/ssh/ssh_config
sudo service ssh start
sudo env >> /etc/environment
while true; do sleep 10; done

