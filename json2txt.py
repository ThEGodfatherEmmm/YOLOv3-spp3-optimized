import os
 
import json
 
 
 
 
 
class BDD_to_YOLOv5:
 
    def __init__(self):
 
        self.writepath = "./bdd100k/labels/100k/train/"
 
        self.bdd100k_width_ratio = 1.0/1280
 
        self.bdd100k_height_ratio = 1.0/720
 
        # self.select_categorys = ["person", "rider", "car", "bus", "truck", "bike","motor", "traffic light", "traffic sign", "train"]
        self.select_categorys = ['bus', 'traffic light', 'traffic sign', 'pedestrian', 'bicycle', 'truck', 'motorcycle', 'car', 'train', 'rider']
        self.categorys_nums = {
 
            "bus": 0,
 
            "traffic light": 1,
 
            "traffic sign": 2,
 
            "pedestrian": 3,
 
            "bicycle": 4,
 
            "truck": 5,
 
            "motorcycle": 6,
 
            "car": 7,
 
            "train": 8,
 
            "rider": 9,
 
        }
 
 
 
    def bdd_to_yolov5(self, path):
 
        lines = ""
 
        with open(path) as fp:
 
            j = json.load(fp)
 
            write = open(self.writepath + "%s.txt" % j["name"], 'w')
 
            for fr in j["frames"]:
 
                for objs in fr["objects"]:
 
                    if objs["category"] in self.select_categorys:
 
                        temp_category=objs["category"]
 
                        idx = self.categorys_nums[temp_category]
 
                        cx = (objs["box2d"]["x1"] + objs["box2d"]["x2"]) / 2.0
 
                        cy = (objs["box2d"]["y1"] + objs["box2d"]["y2"]) / 2.0
 
                        w = objs["box2d"]["x2"] - objs["box2d"]["x1"]
 
                        h = objs["box2d"]["y2"] - objs["box2d"]["y1"]
 
                        if w <= 0 or h <= 0:
 
                            continue
 
                        # 根据图片尺寸进行归一化
 
                        cx, cy, w, h = cx * self.bdd100k_width_ratio, cy * self.bdd100k_height_ratio, w * self.bdd100k_width_ratio, h * self.bdd100k_height_ratio
 
                        line = f"{idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
 
                        lines += line
 
                if len(lines) != 0:
 
                    write.writelines(lines)
 
                    write.close()
 
                    print("%s has been dealt!" % j["name"])
 
 
 
 
 
 
 
if __name__ == "__main__":
 
    bdd_labels_dir = "./bdd100k/labels/100k/train/"
 
    fileList = os.listdir(bdd_labels_dir)
 
    obj = BDD_to_YOLOv5()
 
    for path in fileList:
 
        filepath = bdd_labels_dir+path
 
        print(path)
 
        obj.bdd_to_yolov5(filepath) 