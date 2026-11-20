def get_star_map(file_path):
    map_dict = {}
    with open(file_path, 'r') as f:
        line = f.readline()
        while line:
            pair = line.split()
            map_dict[pair[0]] = pair[1]
            line = f.readline()
    return map_dict

OBJ_CLASSES = ("background", "person", "bag", "bed", "blanket", "book", "box", "broom", "chair", 
           "closet/cabinet", "clothes", "cup/glass/bottle", "dish", "door", 
           "doorknob", "doorway", "floor", "food", "groceries", "laptop", 
           "light", "medicine", "mirror", "paper/notebook", "phone/camera", 
           "picture", "pillow", "refrigerator", "sandwich", "shelf", "shoe", "sofa/couch",
           "table", "television", "towel", "vacuum", "window")

PREDICATE_CLASSES = ("looking at", "not looking at", "unsure", "above", "beneath", "in front of", 
                     "behind", "on the side of", "in", "carrying", "covered by", "drinking from", 
                     "eating", "have it on the back", "holding", "leaning on", "lying on", "not contacting",
                     "other relationship", "sitting on", "standing on", "touching", "twisting", "wearing",
                     "wiping", "writing on")

STAR_PREDICATE_CLASSES = ("looking at", "not looking at", "unsure", "above", "beneath", "in front of", 
                     "behind", "on the side of", "in", "carrying", "covered by", "drinking from", 
                     "eating", "have it on the back", "holding", "leaning on", "lying on", "not contacting",
                     "other relationship", "sitting on", "standing on", "touching", "twisting", "wearing",
                     "wiping", "writing on", "on", "at", "under", "near")