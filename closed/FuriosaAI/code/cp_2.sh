mkdir -p ../compliance/warboy_intel/ssd-mobilenet/SingleStream/TEST01/performance/run_1/
mkdir -p ../compliance/warboy_intel/ssd-mobilenet/SingleStream/TEST04-A/performance/run_1/
mkdir -p ../compliance/warboy_intel/ssd-mobilenet/SingleStream/TEST04-B/performance/run_1/
mkdir -p ../compliance/warboy_intel/ssd-mobilenet/SingleStream/TEST05/performance/run_1/
mkdir -p ../compliance/warboy_intel/ssd-mobilenet/Offline/TEST01/performance/run_1/
mkdir -p ../compliance/warboy_intel/ssd-mobilenet/Offline/TEST04-A/performance/run_1/
mkdir -p ../compliance/warboy_intel/ssd-mobilenet/Offline/TEST04-B/performance/run_1/
mkdir -p ../compliance/warboy_intel/ssd-mobilenet/Offline/TEST05/performance/run_1/

# ssd-mobilenet
cd ssd-mobilenet
mkdir -p build

cp ../inference/compliance/nvidia/TEST01/ssd-mobilenet/audit.config ./
./run-ssd-mobilenet.sh
cp -R build/* ../../compliance/warboy_intel/ssd-mobilenet/SingleStream/TEST01/performance/run_1/

cp ../inference/compliance/nvidia/TEST04-A/audit.config ./
./run-ssd-mobilenet.sh
cp -R build/* ../../compliance/warboy_intel/ssd-mobilenet/SingleStream/TEST04-A/performance/run_1/

cp ../inference/compliance/nvidia/TEST04-B/audit.config ./
./run-ssd-mobilenet.sh
cp -R build/* ../../compliance/warboy_intel/ssd-mobilenet/SingleStream/TEST04-B/performance/run_1/

cp ../inference/compliance/nvidia/TEST05/audit.config ./
./run-ssd-mobilenet.sh
cp -R build/* ../../compliance/warboy_intel/ssd-mobilenet/SingleStream/TEST05/performance/run_1/

rm audit.config
cd -

# ssd-mobilenet
cd ssd-mobilenet
mkdir -p build

cp ../inference/compliance/nvidia/TEST01/ssd-mobilenet/audit.config ./
./run-ssd-mobilenet.sh --scenario Offline
cp -R build/* ../../compliance/warboy_intel/ssd-mobilenet/Offline/TEST01/performance/run_1/

cp ../inference/compliance/nvidia/TEST04-A/audit.config ./
./run-ssd-mobilenet.sh --scenario Offline
./run-ssd-mobilenet.sh --scenario Offline
./run-ssd-mobilenet.sh --scenario Offline
./run-ssd-mobilenet.sh --scenario Offline
cp -R build/* ../../compliance/warboy_intel/ssd-mobilenet/Offline/TEST04-A/performance/run_1/

cp ../inference/compliance/nvidia/TEST04-B/audit.config ./
./run-ssd-mobilenet.sh --scenario Offline
cp -R build/* ../../compliance/warboy_intel/ssd-mobilenet/Offline/TEST04-B/performance/run_1/

cp ../inference/compliance/nvidia/TEST05/audit.config ./
./run-ssd-mobilenet.sh --scenario Offline
cp -R build/* ../../compliance/warboy_intel/ssd-mobilenet/Offline/TEST05/performance/run_1/

rm audit.config
cd -

