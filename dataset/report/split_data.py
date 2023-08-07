import os
import json
import random
root = "Report"

def train():
    with open(os.path.join(root, "train_annotation.json"), "r") as f:
        anno = json.load(f)
    img_list = list(anno.keys())
    with open(os.path.join(root, f"train.txt"),"w") as f:
        for img in img_list:
            f.write(img + ".png" + "\n")

def val_test():
    with open(os.path.join(root, "test_annotation.json"), "r") as f:
        anno = json.load(f)
    img_list = list(anno.keys())
    wsi_list = []
    for i in img_list:
        wsi = i.split("_")[0]
        if wsi not in wsi_list:
            wsi_list.append(wsi)

    for num in range(1, 5):
        random.shuffle(wsi_list)
        print(wsi_list)
        # val_wsi_list = wsi_list[0: len(wsi_list)//3*2]
        # test_wsi_list = wsi_list[len(wsi_list)//3*2 :]
        val_wsi_list = wsi_list[0: len(wsi_list)//2]
        test_wsi_list = wsi_list[len(wsi_list)//2 :]

        val_list, test_list = [], []

        for i in img_list:
            wsi = i.split("_")[0]
            if wsi in val_wsi_list:
                val_list.append(i)
            else:
                test_list.append(i)

        with open(os.path.join(root, f"val_{num}.txt"),"w") as f:
            for img in val_list:
                f.write(img + ".png" + "\n")
        with open(os.path.join(root, f"test_{num}.txt"),"w") as f:
            for img in test_list:
                f.write(img + ".png" + "\n")


if __name__=="__main__":
    train()