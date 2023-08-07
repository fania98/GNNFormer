from dataset.utils import read_json, save_json
import os
def calculate_wsi_nums(file):
    wsis = set()
    with open(file, "r") as f:
        lines = f.readlines()
        print(len(lines))
        for line in lines:
            wsi = line.split("_")[0]
            wsis.add(wsi)
    print(file, len(wsis))


def check_tag(tag_names, sentence, tags):
    valid = False
    for index, t in enumerate(tag_names):
        if t in sentence:
            tags.append(index)
            valid = True
            break

    if not valid:
        tags.append(len(tag_names)-1)

def parse_tags(root, annot_file):
    titles = ["img_name", "hypo", "gt", "confidence_for_high", "pleo", "pleo_gt",
              "crowd", "crowd_gt", "mitosis", "mitosis_gt", "nucleoli", "nucleoli_gt"]

    id_cls = {}

    annot = read_json(os.path.join(root, annot_file ))
    for val in annot:
        if val not in id_cls:
            fields = []
            caption = annot[val]['caption'][0]
            gts = caption.split(".")
            # check_tag(["severe", "moderate", "mild"], gts[0].lower(), fields)
            #
            # check_tag(["severe", "moderate", "mild"], gts[1].lower(), fields)
            #
            # check_tag(["frequent", "infrequent", "rare"], gts[3].lower(), fields)
            #
            # check_tag(["prominent", "inconspicuous"], gts[4].lower(), fields)

            check_tag(["high", "low"], gts[5].lower(), fields)
            id_cls[val] = fields

    save_json(os.path.join(root, annot_file.split("_")[0]+"_cls.json"), id_cls)


parse_tags("Report/", "train_annotation.json")
parse_tags("Report/", "test_annotation.json")

