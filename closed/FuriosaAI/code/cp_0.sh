mkdir -p ../compliance/warboy_intel/resnet50/SingleStream/TEST01/performance/run_1/
mkdir -p ../compliance/warboy_intel/resnet50/SingleStream/TEST04-A/performance/run_1/
mkdir -p ../compliance/warboy_intel/resnet50/SingleStream/TEST04-B/performance/run_1/
mkdir -p ../compliance/warboy_intel/resnet50/SingleStream/TEST05/performance/run_1/

# resnet50
cd resnet50
mkdir -p build

cp ../inference/compliance/nvidia/TEST01/resnet50/audit.config ./
./run-resnet50.sh
cp -R build/* ../../compliance/warboy_intel/resnet50/SingleStream/TEST01/performance/run_1/

cp ../inference/compliance/nvidia/TEST04-A/audit.config ./
./run-resnet50.sh
cp -R build/* ../../compliance/warboy_intel/resnet50/SingleStream/TEST04-A/performance/run_1/

cp ../inference/compliance/nvidia/TEST04-B/audit.config ./
./run-resnet50.sh
cp -R build/* ../../compliance/warboy_intel/resnet50/SingleStream/TEST04-B/performance/run_1/

cp ../inference/compliance/nvidia/TEST05/audit.config ./
./run-resnet50.sh
cp -R build/* ../../compliance/warboy_intel/resnet50/SingleStream/TEST05/performance/run_1/

rm audit.config
cd -
