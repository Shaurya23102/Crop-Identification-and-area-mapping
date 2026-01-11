import cv2
import numpy as np
from ultralytics import SAM
import torch

def generate_cadastral_mask(image_path, output_path):
    model = SAM('mobile_sam.pt')
    results = model.predict(image_path, conf=0.10, iou=0.8, verbose=False)

    # Get the original image dimensions
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        print("Error: Could not load image.")
        return
    h, w = orig_img.shape[:2]

    # 3. Create a blank BLACK canvas
    final_canvas = np.zeros((h, w), dtype=np.uint8)

    # 4. Process Each Detected Mask
    if results[0].masks is not None:
        # Get individual masks
        masks = results[0].masks.data.cpu().numpy()

        for mask in masks:
            # --- FIX FOR YOUR ERROR IS HERE ---
            # Convert Boolean (True/False) to Uint8 (0-255) BEFORE resizing
            mask_uint8_source = mask.astype(np.uint8) * 255

            # Now we can safely resize
            # Use INTER_NEAREST to keep edges sharp/pixelated rather than blurry
            mask_resized = cv2.resize(mask_uint8_source, (w, h), interpolation=cv2.INTER_NEAREST)

            # Find contours for this single field
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                # Filter noise: ignore extremely tiny specks
                if cv2.contourArea(cnt) > 200:
                    # Polygon Approximation for "Surveyor/Map" look
                    # epsilon=0.01 makes lines very straight.
                    # If you want slightly more detail, change 0.01 to 0.005
                    epsilon = 0.001 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)

                    # A. Draw the Field as WHITE (Filled)
                    cv2.drawContours(final_canvas, [approx], -1, 255, thickness=cv2.FILLED)

                    # B. Draw the Boundary as BLACK (Thick Line)
                    # Increased thickness to 4 to define minute boundaries clearly
                    cv2.drawContours(final_canvas, [approx], -1, 0, thickness=4)

    # 5. Save output
    cv2.imwrite(output_path, final_canvas)
    print(f"Success! Mask saved to: {output_path}")

# --- Execution ---
input_file = ''
output_file = ''

generate_cadastral_mask(input_file, output_file)
