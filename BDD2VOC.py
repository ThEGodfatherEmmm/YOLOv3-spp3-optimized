import json
import os

# https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data
# classes = ['bus','light','sign','person','bike','truck','motor','car','train','rider']
classes = ['bus', 'traffic light', 'traffic sign', 'pedestrian', 'bicycle', 'truck', 'motorcycle', 'car', 'train', 'rider']
image_width = 1280
image_height = 720


def convert_bdd_to_voc(filename, folder='labels/det_20/'):
    with open(folder+filename, 'r') as f:
        data = json.load(f)
        cfg = "train/" if "train" in filename else "val/"
        if not os.path.exists(folder+cfg):
            os.mkdir(folder+cfg)
    for image_idx, image_info in enumerate(data):   # 每一张图，data[0], data[1], data[2], ...
        img_name = image_info['name'].strip('.jpg')
        target_file = open(folder + cfg + img_name + '.txt', 'w', encoding='utf-8')
        if 'labels' not in image_info.keys():       # 怎么还有没任何bndbox的图啊。它的.txt文件里就是空的
            continue
        labels = image_info['labels']
        for label_idx, label in enumerate(labels):  # 每一个标签，labels[0], labels[1], ...
            cls = label['category']      # 类别
            if cls not in classes:       # 有三个东西'other vehicle','other person'和'trailer(吊车)'我没把它们算进去，按照10个classes算的
                # print(cls)
                continue
            cls_id = classes.index(cls)
            # xyxy转换为xywh，后者是YOLOv3要求的bndbox格式
            bndbox = label['box2d']
            x1 = float(bndbox['x1'])
            y1 = float(bndbox['y1'])
            x2 = float(bndbox['x2'])
            y2 = float(bndbox['y2'])

            x_center = round(0.5 * (x1 + x2) / image_width, 6)   # 官方教程里保留了6位小数
            y_center = round(0.5 * (y1 + y2) / image_height, 6)
            n_width = round(abs(x2 - x1) / image_width, 6)       # normalized width [0,1]
            n_height = round(abs(y2 - y1) / image_height, 6)     # normalized height [0,1]
            # print(x_center, y_center, n_width, n_height)

            target_file.writelines([str(cls_id)+" "+str(x_center)+" "
                                   +str(y_center)+" "+str(n_width) + " "+str(n_height)+"\n"])

        target_file.close()


if __name__ == "__main__":
    convert_bdd_to_voc('det_val.json')
    convert_bdd_to_voc('det_train.json')