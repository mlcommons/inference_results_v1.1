apt update && \
apt install -y ca-certificates apt-transport-https gnupg && \
apt-key adv --keyserver keyserver.ubuntu.com --recv-key 5F03AFA423A751913F249259814F888B20B09A7E

cat >> /etc/apt/auth.conf.d/furiosa.conf <<EOF
machine archive.furiosa.ai
  login 7f4e0344c9d1473da9de2ce73ddc35e9
  password 3de20dd38d824b84adc5678c3f866ebd
EOF

chmod 400 /etc/apt/auth.conf.d/furiosa.conf

cat > /etc/apt/sources.list.d/furiosa.list << EOF
deb [arch=amd64] https://archive.furiosa.ai/ubuntu focal restricted
EOF

apt-get install -y furiosa-libnpu-u250 furiosa-libnux