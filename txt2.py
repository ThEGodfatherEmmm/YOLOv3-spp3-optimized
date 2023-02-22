import os


if __name__ == "__main__":
 
    bdd_labels_dir = "./bdd100k/labels/val/"
    fileList = os.listdir(bdd_labels_dir)
    writetxt = open('/cluster/home/it_stu103/bw/yolov3/data/bddval.txt', 'w')
    for file in fileList:
        file = file.replace("txt","jpg")
        writetxt.write('/cluster/home/it_stu103/bw/yolo_bdd100k_images/bdd100k/images/100k/val/'+file+'\n')
    writetxt.close()