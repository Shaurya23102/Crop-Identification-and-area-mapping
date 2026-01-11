from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import math
import requests
import base64
from ultralytics import SAM
from PIL import Image
from io import BytesIO
import mercantile

app = Flask(__name__)

# Load model once
model = SAM("mobile_sam.pt")

PATCH_SIZE = 512
ZOOM = 18
TILE_SIZE = 256


def meters_per_pixel(lat, zoom):
    return 156543.03 * math.cos(math.radians(lat)) / (2 ** zoom)


def image_to_base64(img_np):
    _, buffer = cv2.imencode(".png", img_np)
    return base64.b64encode(buffer).decode("utf-8")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect-field", methods=["POST"])
def detect_field():
    lat = request.json["lat"]
    lng = request.json["lng"]

    # --- Get center tile ---
    center_tile = mercantile.tile(lng, lat, ZOOM)

    # --- Stitch 3×3 tiles ---
    canvas = Image.new("RGB", (TILE_SIZE * 3, TILE_SIZE * 3))

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            tile = mercantile.Tile(
                center_tile.x + dx,
                center_tile.y + dy,
                ZOOM
            )

            url = (
                "https://server.arcgisonline.com/ArcGIS/rest/services/"
                f"World_Imagery/MapServer/tile/{ZOOM}/{tile.y}/{tile.x}"
            )

            tile_img = Image.open(
                BytesIO(requests.get(url).content)
            ).convert("RGB")

            canvas.paste(
                tile_img,
                ((dx + 1) * TILE_SIZE, (dy + 1) * TILE_SIZE)
            )

    # --- Crop center patch ---
    half = PATCH_SIZE // 2
    center_px = TILE_SIZE + TILE_SIZE // 2

    patch = canvas.crop((
        center_px - half,
        center_px - half,
        center_px + half,
        center_px + half
    ))

    patch_np = np.array(patch)

    # =====================================================
    # FIX 1 + FIX 2 : MULTI-POINT + NEGATIVE PROMPT
    # =====================================================
    cx = cy = PATCH_SIZE // 2
    offset = 50  # increase if field is very large

    points = [
        [cx, cy],                  # center
        [cx + offset, cy],         # right
        [cx - offset, cy],         # left
        [cx, cy + offset],         # down
        [cx, cy - offset],         # up
        [20, 20]                   # NEGATIVE background point
    ]

    labels = [1, 1, 1, 1, 1, 0]

    results = model.predict(
        patch_np,
        points=points,
        labels=labels,
        conf=0.1,
        verbose=False
    )

    mask = results[0].masks.data[0].cpu().numpy()

    # --- Binary mask (white = field) ---
    binary_mask = (mask > 0).astype(np.uint8) * 255

    # --- AREA CALCULATION (PIXEL-BASED) ---
    white_pixels = np.sum(binary_mask == 255)
    mpp = meters_per_pixel(lat, ZOOM)
    area_hectares = (white_pixels * (mpp ** 2)) / 10_000

    # --- Boundary ONLY for visualization ---
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cnt = max(contours, key=cv2.contourArea)
    epsilon = 0.0001 * cv2.arcLength(cnt, True)
    poly = cv2.approxPolyDP(cnt, epsilon, True)
    boundary_pixels = poly.squeeze().tolist()

    return jsonify({
        "area": area_hectares,
        "patch_image": image_to_base64(patch_np),
        "mask_image": image_to_base64(binary_mask),
        "boundary_pixels": boundary_pixels,
        "patch_size": PATCH_SIZE,
        "center": {"lat": lat, "lng": lng},
        "zoom": ZOOM
    })


if __name__ == "__main__":
    app.run(debug=True)
