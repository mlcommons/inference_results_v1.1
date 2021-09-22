mkdir -p ../compliance/warboy_intel/ssd-resnet34/SingleStream/TEST01/performance/run_1/
mkdir -p ../compliance/warboy_intel/ssd-resnet34/SingleStream/TEST04-A/performance/run_1/
mkdir -p ../compliance/warboy_intel/ssd-resnet34/SingleStream/TEST04-B/performance/run_1/
mkdir -p ../compliance/warboy_intel/ssd-resnet34/SingleStream/TEST05/performance/run_1/
mkdir -p ../compliance/warboy_intel/ssd-resnet34/Offline/TEST01/performance/run_1/
mkdir -p ../compliance/warboy_intel/ssd-resnet34/Offline/TEST04-A/performance/run_1/
mkdir -p ../compliance/warboy_intel/ssd-resnet34/Offline/TEST04-B/performance/run_1/
mkdir -p ../compliance/warboy_intel/ssd-resnet34/Offline/TEST05/performance/run_1/


# ssd-resnet34
cd ssd-resnet34
mkdir -p build

cp ../inference/compliance/nvidia/TEST01/ssd-resnet34/audit.config ./
./run-ssd-resnet34.sh
cp -R build/* ../../compliance/warboy_intel/ssd-resnet34/SingleStream/TEST01/performance/run_1/

cp ../inference/compliance/nvidia/TEST04-A/audit.config ./
./run-ssd-resnet34.sh
cp -R build/* ../../compliance/warboy_intel/ssd-resnet34/SingleStream/TEST04-A/performance/run_1/

cp ../inference/compliance/nvidia/TEST04-B/audit.config ./
./run-ssd-resnet34.sh
cp -R build/* ../../compliance/warboy_intel/ssd-resnet34/SingleStream/TEST04-B/performance/run_1/

cp ../inference/compliance/nvidia/TEST05/audit.config ./
./run-ssd-resnet34.sh
cp -R build/* ../../compliance/warboy_intel/ssd-resnet34/SingleStream/TEST05/performance/run_1/

rm audit.config
cd -

# ssd-resnet34
cd ssd-resnet34
mkdir -p build

cp ../inference/compliance/nvidia/TEST01/ssd-resnet34/audit.config ./
./run-ssd-resnet34.sh --scenario Offline
cp -R build/* ../../compliance/warboy_intel/ssd-resnet34/Offline/TEST01/performance/run_1/

cp ../inference/compliance/nvidia/TEST04-A/audit.config ./
./run-ssd-resnet34.sh --scenario Offline
cp -R build/* ../../compliance/warboy_intel/ssd-resnet34/Offline/TEST04-A/performance/run_1/

cp ../inference/compliance/nvidia/TEST04-B/audit.config ./
./run-ssd-resnet34.sh --scenario Offline
cp -R build/* ../../compliance/warboy_intel/ssd-resnet34/Offline/TEST04-B/performance/run_1/

cp ../inference/compliance/nvidia/TEST05/audit.config ./
./run-ssd-resnet34.sh --scenario Offline
cp -R build/* ../../compliance/warboy_intel/ssd-resnet34/Offline/TEST05/performance/run_1/

rm audit.config
cd -
