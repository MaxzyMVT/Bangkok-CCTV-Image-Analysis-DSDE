import sys
from pathlib import Path

# Add project root and Models
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "Models"))

from Models.detector import YOLOVehicleDetector
from Models.pipeline import parse_cctv_filename

def main():
    print("Testing YOLO Detection Pipeline...")
    detector = YOLOVehicleDetector(model_name="yolov8n.pt", confidence_threshold=0.25)
    
    # Pick a sample image from dataset
    dataset_dir = ROOT_DIR / "dataset" / "100_cameras" / "cctv_images"
    sample_images = list(dataset_dir.rglob("*.jpg"))
    if not sample_images:
        print("No images found in dataset folder.")
        return
    
    test_img = sample_images[0]
    print(f"Sample test image: {test_img.name}")
    parsed = parse_cctv_filename(test_img.name)
    print(f"Parsed metadata: {parsed}")
    
    # Run prediction
    result = detector.predict(test_img, draw_annotations=True)
    counts = result["counts"]
    print("\n--- Detection Results ---")
    print(f"Total Vehicles: {counts['total_vehicles']}")
    for k, v in counts.items():
        if k != "total_vehicles" and v > 0:
            print(f"  - {k}: {v}")
            
    print(f"Number of detections: {len(result['detections'])}")
    
    # Save test output
    output_path = ROOT_DIR / "Models" / "test_annotated_sample.jpg"
    from Models.detector import imwrite_unicode
    imwrite_unicode(output_path, result["annotated_image"])
    print(f"\n[+] Annotated sample saved to: {output_path}")

if __name__ == "__main__":
    main()
