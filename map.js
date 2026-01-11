const map = L.map('map');

// Satellite imagery
L.tileLayer(
 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
 { maxZoom: 19 }
).addTo(map);

map.setMinZoom(17);
map.setMaxZoom(19);

let fieldLayer = null;

// Auto locate
map.locate({ setView: true, maxZoom: 17 });

map.on('locationerror', () => {
    map.setView([20.5937, 78.9629], 5);
});

// Convert patch pixel → lat/lng
function patchPixelToLatLng(px, py, patchSize, center, zoom) {
    const mpp = 156543.03 * Math.cos(center.lat * Math.PI / 180) / (2 ** zoom);

    const dx = px - patchSize / 2;
    const dy = py - patchSize / 2;

    const dLng = (dx * mpp) / (111320 * Math.cos(center.lat * Math.PI / 180));
    const dLat = -(dy * mpp) / 110540;

    return [center.lat + dLat, center.lng + dLng];
}

map.on('click', async (e) => {
    if (fieldLayer) map.removeLayer(fieldLayer);

    const res = await fetch('/detect-field', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            lat: e.latlng.lat,
            lng: e.latlng.lng
        })
    });

    const data = await res.json();

    // Show debug panel
    document.getElementById('debug').style.display = 'block';
    document.getElementById('patch-img').src =
        "data:image/png;base64," + data.patch_image;
    document.getElementById('mask-img').src =
        "data:image/png;base64," + data.mask_image;
    document.getElementById('area-text').innerHTML =
        `<b>Area:</b> ${data.area.toFixed(2)} hectares`;

    // Draw boundary
    const boundary = data.boundary_pixels.map(p =>
        patchPixelToLatLng(
            p[0], p[1],
            data.patch_size,
            data.center,
            data.zoom
        )
    );

    fieldLayer = L.polygon(boundary, {
        color: '#00ff00',
        weight: 3,
        fillOpacity: 0.25
    }).addTo(map);
});
