import os
import xml.etree.ElementTree as ET

# Define paths
VOC_ANNOTATIONS_PATH = "/home/andy/datasets/apples_voc/Annotations"
YOLO_LABELS_PATH = "/home/andy/datasets/apples_voc/labels"  # Create a folder named "labels"
IMAGE_PATH = "/home/andy/datasets/apples_voc/JPEGImages"

# Class list (Modify this with your actual class names)
CLASS_NAMES = ["apple"]  # Replace with your actual classes

# Create labels directory if it doesn't exist
os.makedirs(YOLO_LABELS_PATH, exist_ok=True)

def convert_voc_to_yolo(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Get image size
    size = root.find("size")
    img_width = int(size.find("width").text)
    img_height = int(size.find("height").text)

    yolo_lines = []
    
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name not in CLASS_NAMES:
            continue  # Skip unknown classes
        
        class_id = CLASS_NAMES.index(class_name)
        
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # Convert to YOLO format (normalized)
        x_center = (xmin + xmax) / (2.0 * img_width)
        y_center = (ymin + ymax) / (2.0 * img_height)
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Save YOLO format labels
    txt_filename = os.path.join(YOLO_LABELS_PATH, os.path.basename(xml_file).replace(".xml", ".txt"))
    with open(txt_filename, "w") as f:
        f.write("\n".join(yolo_lines))

# Convert all annotations
for xml_file in os.listdir(VOC_ANNOTATIONS_PATH):
    if xml_file.endswith(".xml"):
        convert_voc_to_yolo(os.path.join(VOC_ANNOTATIONS_PATH, xml_file))

print("Conversion completed! YOLO labels are saved in:", YOLO_LABELS_PATH)
