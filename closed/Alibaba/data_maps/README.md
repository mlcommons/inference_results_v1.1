# Data Maps

Below shows the steps to get these data maps:

- `coco/val_map.txt`: Run the following Python script:
```
import json
with open("build/data/coco/annotations/instances_val2017.json") as f:
    annotations = json.load(f)
with open("data_maps/coco/val_map.txt", "w") as f:
    print("\n".join([i["file_name"] for i in annotations["images"]]), file=f)
```
- `coco/cal_map.txt`: Downloaded from [coco_cal_images_list.txt](https://github.com/mlperf/inference/blob/master/calibration/COCO/coco_cal_images_list.txt) in the reference repository.
