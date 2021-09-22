mkdir -p ../compliance/warboy_intel/resnet50/Offline/TEST01/results/
mkdir -p ../compliance/warboy_intel/resnet50/Offline/TEST04-A/results/
mkdir -p ../compliance/warboy_intel/resnet50/Offline/TEST04-B/results/
mkdir -p ../compliance/warboy_intel/resnet50/Offline/TEST05/results/

mkdir -p ../compliance/warboy_intel/ssd-mobilenet/Offline/TEST01/results/
mkdir -p ../compliance/warboy_intel/ssd-mobilenet/Offline/TEST04-A/results/
mkdir -p ../compliance/warboy_intel/ssd-mobilenet/Offline/TEST04-B/results/
mkdir -p ../compliance/warboy_intel/ssd-mobilenet/Offline/TEST05/results/

mkdir -p ../compliance/warboy_intel/ssd-resnet34/Offline/TEST01/results/
mkdir -p ../compliance/warboy_intel/ssd-resnet34/Offline/TEST04-A/results/
mkdir -p ../compliance/warboy_intel/ssd-resnet34/Offline/TEST04-B/results/
mkdir -p ../compliance/warboy_intel/ssd-resnet34/Offline/TEST05/results/


# resnet50
cd resnet50
mkdir -p build

cp ../inference/compliance/nvidia/TEST01/resnet50/audit.config ./
./run-resnet50.sh --scenario Offline
cp -R build/* ../../compliance/warboy_intel/resnet50/Offline/TEST01/results/

cp ../inference/compliance/nvidia/TEST04-A/audit.config ./
./run-resnet50.sh --scenario Offline
cp -R build/* ../../compliance/warboy_intel/resnet50/Offline/TEST04-A/results/

cp ../inference/compliance/nvidia/TEST04-B/audit.config ./
./run-resnet50.sh --scenario Offline
cp -R build/* ../../compliance/warboy_intel/resnet50/Offline/TEST04-B/results/

cp ../inference/compliance/nvidia/TEST05/audit.config ./
./run-resnet50.sh --scenario Offline
cp -R build/* ../../compliance/warboy_intel/resnet50/Offline/TEST05/results/

rm audit.config
cd -

# ssd-mobilenet
cd ssd-mobilenet
mkdir -p build

cp ../inference/compliance/nvidia/TEST01/ssd-mobilenet/audit.config ./
./run-ssd-mobilenet.sh --scenario Offline
cp -R build/* ../../compliance/warboy_intel/ssd-mobilenet/Offline/TEST01/results/

cp ../inference/compliance/nvidia/TEST04-A/audit.config ./
./run-ssd-mobilenet.sh --scenario Offline
./run-ssd-mobilenet.sh --scenario Offline
./run-ssd-mobilenet.sh --scenario Offline
./run-ssd-mobilenet.sh --scenario Offline
cp -R build/* ../../compliance/warboy_intel/ssd-mobilenet/Offline/TEST04-A/results/

cp ../inference/compliance/nvidia/TEST04-B/audit.config ./
./run-ssd-mobilenet.sh --scenario Offline
cp -R build/* ../../compliance/warboy_intel/ssd-mobilenet/Offline/TEST04-B/results/

cp ../inference/compliance/nvidia/TEST05/audit.config ./
./run-ssd-mobilenet.sh --scenario Offline
cp -R build/* ../../compliance/warboy_intel/ssd-mobilenet/Offline/TEST05/results/

rm audit.config
cd -

# ssd-resnet34
cd ssd-resnet34
mkdir -p build

cp ../inference/compliance/nvidia/TEST01/ssd-resnet34/audit.config ./
./run-ssd-resnet34.sh --scenario Offline
cp -R build/* ../../compliance/warboy_intel/ssd-resnet34/Offline/TEST01/results/

cp ../inference/compliance/nvidia/TEST04-A/audit.config ./
./run-ssd-resnet34.sh --scenario Offline
cp -R build/* ../../compliance/warboy_intel/ssd-resnet34/Offline/TEST04-A/results/

cp ../inference/compliance/nvidia/TEST04-B/audit.config ./
./run-ssd-resnet34.sh --scenario Offline
cp -R build/* ../../compliance/warboy_intel/ssd-resnet34/Offline/TEST04-B/results/

cp ../inference/compliance/nvidia/TEST05/audit.config ./
./run-ssd-resnet34.sh --scenario Offline
cp -R build/* ../../compliance/warboy_intel/ssd-resnet34/Offline/TEST05/results/

rm audit.config
cd -
