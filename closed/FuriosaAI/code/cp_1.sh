mkdir -p ../compliance/warboy_intel/resnet50/Offline/TEST01/performance/run_1/
mkdir -p ../compliance/warboy_intel/resnet50/Offline/TEST04-A/performance/run_1/
mkdir -p ../compliance/warboy_intel/resnet50/Offline/TEST04-B/performance/run_1/
mkdir -p ../compliance/warboy_intel/resnet50/Offline/TEST05/performance/run_1/

# resnet50
cd resnet50
mkdir -p build

cp ../inference/compliance/nvidia/TEST01/resnet50/audit.config ./
./run-resnet50.sh --scenario Offline
cp -R build/* ../../compliance/warboy_intel/resnet50/Offline/TEST01/performance/run_1/

cp ../inference/compliance/nvidia/TEST04-A/audit.config ./
./run-resnet50.sh --scenario Offline
cp -R build/* ../../compliance/warboy_intel/resnet50/Offline/TEST04-A/performance/run_1/

cp ../inference/compliance/nvidia/TEST04-B/audit.config ./
./run-resnet50.sh --scenario Offline
cp -R build/* ../../compliance/warboy_intel/resnet50/Offline/TEST04-B/performance/run_1/

cp ../inference/compliance/nvidia/TEST05/audit.config ./
./run-resnet50.sh --scenario Offline
cp -R build/* ../../compliance/warboy_intel/resnet50/Offline/TEST05/performance/run_1/

rm audit.config
cd -
