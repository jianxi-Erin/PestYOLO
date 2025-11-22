import os
import xml.etree.ElementTree as ET
import shutil
from PIL import Image
import re

# -------------------------------
# 配置路径
# -------------------------------
dataset_root = "IP102_v1.1/Detection/VOC2007"
images_dir = os.path.join(dataset_root, "JPEGImages")
ann_dir = os.path.join(dataset_root, "Annotations")
sets_dir = os.path.join(dataset_root, "ImageSets", "Main")  # train.txt / test.txt

output_root = "IP102_YOLOv8"
train_dir = os.path.join(output_root, "train")
val_dir = os.path.join(output_root, "val")

# -------------------------------
# 官方 classes.txt 102 类（英文名称）
# -------------------------------


classes_name = [
    "rice leaf roller","rice leaf caterpillar","paddy stem maggot","asiatic rice borer",
    "yellow rice borer","rice gall midge","Rice Stemfly","brown plant hopper",
    "white backed plant hopper","small brown plant hopper","rice water weevil",
    "rice leafhopper","grain spreader thrips","rice shell pest","grub","mole cricket",
    "wireworm","white margined moth","black cutworm","large cutworm","yellow cutworm",
    "red spider","corn borer","army worm","aphids","Potosiabre vitarsis","peach borer",
    "english grain aphid","green bug","bird cherry-oataphid","wheat blossom midge",
    "penthaleus major","longlegged spider mite","wheat phloeothrips","wheat sawfly",
    "cerodonta denticornis","beet fly","flea beetle","cabbage army worm","beet army worm",
    "Beet spot flies","meadow moth","beet weevil","sericaorient alismots chulsky",
    "alfalfa weevil","flax budworm","alfalfa plant bug","tarnished plant bug",
    "Locustoidea","lytta polita","legume blister beetle","blister beetle",
    "therioaphis maculata Buckton","odontothrips loti","Thrips","alfalfa seed chalcid",
    "Pieris canidia","Apolygus lucorum","Limacodidae","Viteus vitifoliae","Colomerus vitis",
    "Brevipoalpus lewisi McGregor","oides decempunctata","Polyphagotars onemus latus",
    "Pseudococcus comstocki Kuwana","parathrene regalis","Ampelophaga","Lycorma delicatula",
    "Xylotrechus","Cicadella viridis","Miridae","Trialeurodes vaporariorum","Erythroneura apicalis",
    "Papilio xuthus","Panonchus citri McGregor","Phyllocoptes oleiverus ashmead",
    "Icerya purchasi Maskell","Unaspis yanonensis","Ceroplastes rubens","Chrysomphalus aonidum",
    "Parlatoria zizyphus Lucus","Nipaecoccus vastalor","Aleurocanthus spiniferus",
    "Tetradacus c Bactrocera minax","Dacus dorsalis(Hendel)","Bactrocera tsuneonis",
    "Prodenia litura","Adristyrannus","Phyllocnistis citrella Stainton",
    "Toxoptera citricidus","Toxoptera aurantii","Aphis citricola Vander Goot",
    "Scirtothrips dorsalis Hood","Dasineura sp","Lawana imitata Melichar",
    "Salurnis marginella Guerr","Deporaus marginatus Pascoe","Chlumetia transversa",
    "Mango flat beak leafhopper","Rhytidodera bowrinii white","Sternochetus frigidus",
    "Cicadellidae"
]
# 中文名
# classes_name = [
#     "稻纵卷叶螟", "稻叶毛虫", "稻秆潜蝇", "亚洲稻螟",
#     "黄稻螟", "稻瘿蚊", "稻秆蝇", "褐飞虱",
#     "白背飞虱", "灰飞虱", "稻水象甲",
#     "稻叶蝉", "禾谷蓟马", "稻壳害虫", "蛴螬", "蝼蛄",
#     "金针虫", "白缘蛾", "黑切根虫", "大切根虫", "黄切根虫",
#     "红蜘蛛", "玉米螟", "黏虫", "蚜虫", "葡萄金斑蛾", "桃蛀螟",
#     "麦长管蚜", "绿bug", "燕麦蚜", "麦穗蛾",
#     "麦大赤螨", "长腿红蜘蛛", "麦皮蓟马", "麦叶蜂",
#     "齿角蝇", "甜菜潜蝇", "跳甲", "菜青虫", "甜菜夜蛾",
#     "甜菜斑蝇", "草地螟", "甜菜象甲", "东方绢金龟",
#     "苜蓿叶象甲", "亚麻芽蛾", "苜蓿盲蝽", "牧草盲蝽",
#     "蝗总科", "绿芫菁", "豆芫菁", "斑芫菁",
#     "苜蓿斑蚜", "豆蓟马", "蓟马", "苜蓿籽蜂",
#     "东方菜粉蝶", "绿盲蝽", "刺蛾科", "葡萄根瘤蚜", "葡萄瘿螨",
#     "刘氏短须螨", "十星瓢萤叶甲", "茶黄螨",
#     "康氏粉蚧", "葡萄透翅蛾", "葡萄天蛾", "斑衣蜡蝉",
#     "葡萄虎天牛", "大青叶蝉", "盲蝽科", "温室粉虱", "葡萄斑叶蝉",
#     "柑橘凤蝶", "柑橘全爪螨", "柑橘锈螨",
#     "吹绵蚧", "矢尖蚧", "红蜡蚧", "褐圆蚧",
#     "黑点蚧", "橘鳞粉蚧", "黑刺粉虱",
#     "柑橘小实蝇", "橘大实蝇", "蜜柑大实蝇",
#     "斜纹夜蛾", "黄斑卷叶蛾", "柑橘潜叶蛾",
#     "柑橘木虱", "橘二叉蚜", "橘蚜",
#     "茶黄蓟马", "柑橘花蕾蛆", "碧蛾蜡蝉",
#     "碧蜡蝉", "芒果切叶象甲", "柑橘黑线麦蛾",
#     "芒果扁喙叶蝉", "芒果脊胸天牛", "芒果果核象甲",
#     "叶蝉科"
# ]

# 保存 classes.txt
os.makedirs(output_root, exist_ok=True)
with open(os.path.join(output_root, "classes.txt"), "w", encoding="utf-8") as f:
    for cls in classes_name:
        f.write(cls + "\n")

# -------------------------------
# 修复 XML 函数（保留第一个 <annotation> ... </annotation>）
# -------------------------------
def clean_xml(xml_file):
    with open(xml_file, "r", encoding="utf-8") as f:
        content = f.read()
    match = re.search(r"<annotation>.*</annotation>", content, re.DOTALL)
    if match:
        with open(xml_file, "w", encoding="utf-8") as f:
            f.write(match.group(0))
        return True
    return False

# -------------------------------
# YOLO 标签转换函数
# -------------------------------
def convert_xml_to_yolo(xml_file, img_w, img_h):
    try:
        tree = ET.parse(xml_file)
    except ET.ParseError:
        print(f"❌ XML 解析失败，尝试修复: {xml_file}")
        if not clean_xml(xml_file):
            print(f"❌ 修复失败，跳过: {xml_file}")
            return []
        tree = ET.parse(xml_file)
    root = tree.getroot()
    yolo_boxes = []
    for obj in root.findall("object"):
        cls_id = int(obj.find("name").text)
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        bw = (xmax - xmin) / img_w
        bh = (ymax - ymin) / img_h
        yolo_boxes.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")
    return yolo_boxes

# -------------------------------
# 创建输出目录
# -------------------------------
for d in [train_dir, val_dir]:
    os.makedirs(os.path.join(d, "images"), exist_ok=True)
    os.makedirs(os.path.join(d, "labels"), exist_ok=True)

# -------------------------------
# 加载 train/test 列表
# -------------------------------
def load_image_list(txt_file):
    with open(txt_file, "r") as f:
        return [line.strip() + ".jpg" for line in f.readlines()]

train_list = load_image_list(os.path.join(sets_dir, "trainval.txt"))
val_list   = load_image_list(os.path.join(sets_dir, "test.txt"))

# -------------------------------
# 生成 YOLO 标签
# -------------------------------
for img_set, split_dir in zip([train_list, val_list], [train_dir, val_dir]):
    for img_name in img_set:
        img_path = os.path.join(images_dir, img_name)
        xml_path = os.path.join(ann_dir, img_name.replace(".jpg", ".xml"))
        if not os.path.exists(xml_path):
            print(f"⚠️ XML 文件不存在: {xml_path}")
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        yolo_boxes = convert_xml_to_yolo(xml_path, w, h)
        if not yolo_boxes:
            continue
        label_path = os.path.join(split_dir, "labels", img_name.replace(".jpg", ".txt"))
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_boxes))
        shutil.copy(img_path, os.path.join(split_dir, "images", img_name))

# -------------------------------
# 生成 data.yaml
# -------------------------------
data_yaml = f"""
train: {train_dir}/images
val: {val_dir}/images

nc: {len(classes_name)}
names: {classes_name}
"""

with open(os.path.join(output_root, "data.yaml"), "w", encoding="utf-8") as f:
    f.write(data_yaml.strip())

print(f"✅ 数据集已处理为 YOLO 格式 → {output_root}")
