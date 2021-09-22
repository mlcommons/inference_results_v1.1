mkdir -p ../compliance/warboy_intel/resnet50/SingleStream/TEST01/results/
mkdir -p ../compliance/warboy_intel/resnet50/SingleStream/TEST04-A/results//
mkdir -p ../compliance/warboy_intel/resnet50/SingleStream/TEST04-B/results/
mkdir -p ../compliance/warboy_intel/resnet50/SingleStream/TEST05/results/

mkdir -p ../compliance/warboy_intel/ssd-mobilenet/SingleStream/TEST01/results/
mkdir -p ../compliance/warboy_intel/ssd-mobilenet/SingleStream/TEST04-A/results/
mkdir -p ../compliance/warboy_intel/ssd-mobilenet/SingleStream/TEST04-B/results/
mkdir -p ../compliance/warboy_intel/ssd-mobilenet/SingleStream/TEST05/results/

mkdir -p ../compliance/warboy_intel/ssd-resnet34/SingleStream/TEST01/results/
mkdir -p ../compliance/warboy_intel/ssd-resnet34/SingleStream/TEST04-A/results/
mkdir -p ../compliance/warboy_intel/ssd-resnet34/SingleStream/TEST04-B/results/
mkdir -p ../compliance/warboy_intel/ssd-resnet34/SingleStream/TEST05/results/


# resnet50
cd resnet50
mkdir -p build

cp ../inference/compliance/nvidia/TEST01/resnet50/audit.config ./
./run-resnet50.sh
cp -R build/* ../../compliance/warboy_intel/resnet50/SingleStream/TEST01/results/

cp ../inference/compliance/nvidia/TEST04-A/audit.config ./
./run-resnet50.sh
cp -R build/* ../../compliance/warboy_intel/resnet50/SingleStream/TEST04-A/results/

cp ../inference/compliance/nvidia/TEST04-B/audit.config ./
./run-resnet50.sh
cp -R build/* ../../compliance/warboy_intel/resnet50/SingleStream/TEST04-B/results/

cp ../inference/compliance/nvidia/TEST05/audit.config ./
./run-resnet50.sh
cp -R build/* ../../compliance/warboy_intel/resnet50/SingleStream/TEST05/results/

rm audit.config
cd -

# ssd-mobilenet
cd ssd-mobilenet
mkdir -p build

cp ../inference/compliance/nvidia/TEST01/ssd-mobilenet/audit.config ./
./run-ssd-mobilenet.sh
cp -R build/* ../../compliance/warboy_intel/ssd-mobilenet/SingleStream/TEST01/results/

cp ../inference/compliance/nvidia/TEST04-A/audit.config ./
./run-ssd-mobilenet.sh
cp -R build/* ../../compliance/warboy_intel/ssd-mobilenet/SingleStream/TEST04-A/results/

cp ../inference/compliance/nvidia/TEST04-B/audit.config ./
./run-ssd-mobilenet.sh
cp -R build/* ../../compliance/warboy_intel/ssd-mobilenet/SingleStream/TEST04-B/results/

cp ../inference/compliance/nvidia/TEST05/audit.config ./
./run-ssd-mobilenet.sh
cp -R build/* ../../compliance/warboy_intel/ssd-mobilenet/SingleStream/TEST05/results/

rm audit.config
cd -

# ssd-resnet34
cd ssd-resnet34
mkdir -p build

cp ../inference/compliance/nvidia/TEST01/ssd-resnet34/audit.config ./
./run-ssd-resnet34.sh
cp -R build/* ../../compliance/warboy_intel/ssd-resnet34/SingleStream/TEST01/results/

cp ../inference/compliance/nvidia/TEST04-A/audit.config ./
./run-ssd-resnet34.sh
cp -R build/* ../../compliance/warboy_intel/ssd-resnet34/SingleStream/TEST04-A/results/

cp ../inference/compliance/nvidia/TEST04-B/audit.config ./
./run-ssd-resnet34.sh
cp -R build/* ../../compliance/warboy_intel/ssd-resnet34/SingleStream/TEST04-B/results/

cp ../inference/compliance/nvidia/TEST05/audit.config ./
./run-ssd-resnet34.sh
cp -R build/* ../../compliance/warboy_intel/ssd-resnet34/SingleStream/TEST05/results/

rm audit.config
cd -
