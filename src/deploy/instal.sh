curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo \
  "deb [arch=arm64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io
usermod -aG docker rock64
docker version

docker swarm join --token SWMTKN-1-4t6n677mik3uwufmin4puvd1fl3h8zqs8lke6oc013ac5wb3z4-2ztcqxjjfaie4i5pgqprvk8z0 192.168.88.252:2377
