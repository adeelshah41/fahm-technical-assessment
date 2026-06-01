import os
import csv
import random

data_root = r"D:\fahm\model\data"
jpeg_dir = os.path.join(data_root, "jpeg")
csv_dir = os.path.join(data_root, "csv")

def main():
    print("Scanning local JPEG folders and cross-referencing with CSVs...")
    if not os.path.exists(jpeg_dir):
        print(f"Error: JPEG directory not found at {jpeg_dir}")
        return
        
    local_uids = set(os.listdir(jpeg_dir))

    # Build UID -> Cropped Image filename mapping from dicom_info.csv
    uid_to_image = {}
    dicom_path = os.path.join(csv_dir, "dicom_info.csv")
    
    if not os.path.exists(dicom_path):
        print("Error: dicom_info.csv not found.")
        return
        
    with open(dicom_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row.get("SeriesInstanceUID", "").strip()
            desc = row.get("SeriesDescription", "").strip().lower()
            img_path = row.get("image_path", "").strip()
            
            if uid in local_uids and "cropped images" in desc:
                # Extract just the filename (e.g. 1-272.jpg)
                filename = img_path.replace("\\", "/").split("/")[-1]
                uid_to_image[uid] = filename

    # Read case descriptions and map UID -> Pathology
    benign_cases = []
    malignant_cases = []

    case_files = [
        "mass_case_description_train_set.csv",
        "mass_case_description_test_set.csv",
        "calc_case_description_train_set.csv",
        "calc_case_description_test_set.csv"
    ]

    for case_file in case_files:
        path = os.path.join(csv_dir, case_file)
        if not os.path.exists(path):
            continue
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pathology = row.get("pathology", "").strip().upper()
                crop_path = row.get("cropped image file path", "").strip()
                
                # The UID is the second to last segment
                parts = crop_path.strip().strip('"').replace("\\", "/").split("/")
                if len(parts) >= 3:
                    uid = parts[-2]
                    if uid in uid_to_image:
                        image_filename = uid_to_image[uid]
                        full_path = os.path.join(jpeg_dir, uid, image_filename)
                        # Verify the file actually exists on disk
                        if os.path.exists(full_path):
                            if "BENIGN" in pathology:
                                benign_cases.append(full_path)
                            elif "MALIGNANT" in pathology:
                                malignant_cases.append(full_path)

    # Deduplicate and shuffle
    benign_cases = list(set(benign_cases))
    malignant_cases = list(set(malignant_cases))

    random.shuffle(benign_cases)
    random.shuffle(malignant_cases)

    print("\n" + "="*60)
    print(f"FOUND {len(benign_cases)} LOCAL BENIGN IMAGES. HERE ARE 10 RANDOM ONES:")
    print("="*60)
    for p in benign_cases[:10]:
        print(p)
        
    print("\n" + "="*60)
    print(f"FOUND {len(malignant_cases)} LOCAL MALIGNANT IMAGES. HERE ARE 10 RANDOM ONES:")
    print("="*60)
    for p in malignant_cases[:10]:
        print(p)
        
    print("\nDone! You can drag and drop any of these file paths directly into the UI.")

if __name__ == "__main__":
    main()
