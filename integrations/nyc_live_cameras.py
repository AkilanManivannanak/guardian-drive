"""
Guardian Drive™ v4.2 — Live NYC Traffic Camera Integration
Uses NYC DOT public camera API (964 cameras) to provide
real-time traffic camera feeds based on current GPS location.

Replaces static nuScenes images with live NYC camera feeds.
Cameras selected by proximity to ego vehicle location.
"""
from __future__ import annotations
import urllib.request, json, base64, math, time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class NYCCamera:
    id:        str
    name:      str
    lat:       float
    lon:       float
    area:      str
    is_online: bool
    image_url: str
    distance_m: float = 0.0


class NYCLiveCameraFeed:
    """
    Streams live NYC DOT traffic camera images to Guardian Drive.
    Selects 6 nearest cameras to current GPS location.
    Updates every window call with fresh images.
    """

    API_URL = "https://webcams.nyctmc.org/api/cameras"
    CACHE_TTL = 300  # refresh camera list every 5 min
    IMG_TTL   = 5    # refresh images every 5 sec

    # Camera roles matching nuScenes layout
    CAM_ROLES = [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    def __init__(self):
        self._cameras:     List[NYCCamera] = []
        self._cache_time:  float = 0.0
        self._img_cache:   Dict[str, tuple] = {}  # id → (b64, timestamp)
        self._last_lat:    float = 40.6782  # Brooklyn default
        self._last_lon:    float = -73.9442
        self._selected:    List[NYCCamera] = []
        self._n_fetched:   int = 0

        # Load cameras immediately
        self._refresh_camera_list()
        print(f"✓ NYC Live Cameras: {len(self._cameras)} cameras loaded")

    def _haversine(self, lat1, lon1, lat2, lon2) -> float:
        """Distance in meters."""
        R = 6371000
        dlat = math.radians(lat2-lat1)
        dlon = math.radians(lon2-lon1)
        a = (math.sin(dlat/2)**2 +
             math.cos(math.radians(lat1)) *
             math.cos(math.radians(lat2)) *
             math.sin(dlon/2)**2)
        return R * 2 * math.asin(math.sqrt(a))

    def _refresh_camera_list(self):
        """Fetch all NYC DOT cameras."""
        try:
            req = urllib.request.Request(
                self.API_URL,
                headers={'User-Agent': 'GuardianDrive/4.2'})
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read())
            self._cameras = [
                NYCCamera(
                    id        = c['id'],
                    name      = c['name'],
                    lat       = float(c['latitude']),
                    lon       = float(c['longitude']),
                    area      = c.get('area','NYC'),
                    is_online = c.get('isOnline','true') == 'true',
                    image_url = c['imageUrl'],
                )
                for c in data
                if c.get('isOnline','true') == 'true'
            ]
            self._cache_time = time.time()
        except Exception as e:
            print(f"⚠ NYC camera list refresh failed: {e}")

    def select_nearest(self, lat: float, lon: float,
                       n: int = 6) -> List[NYCCamera]:
        """Select n nearest online cameras to GPS location."""
        now = time.time()
        if now - self._cache_time > self.CACHE_TTL:
            self._refresh_camera_list()

        # Calculate distances
        for cam in self._cameras:
            cam.distance_m = self._haversine(lat, lon, cam.lat, cam.lon)

        # Sort by distance, return n nearest
        sorted_cams = sorted(self._cameras, key=lambda c: c.distance_m)
        return sorted_cams[:n]

    def _fetch_image(self, cam: NYCCamera) -> str:
        """Fetch camera image as base64. Uses cache."""
        now = time.time()
        cached = self._img_cache.get(cam.id)
        if cached and now - cached[1] < self.IMG_TTL:
            return cached[0]

        try:
            req = urllib.request.Request(
                cam.image_url,
                headers={'User-Agent': 'GuardianDrive/4.2'})
            with urllib.request.urlopen(req, timeout=4) as r:
                img_bytes = r.read()
            b64 = base64.b64encode(img_bytes).decode()
            self._img_cache[cam.id] = (b64, now)
            self._n_fetched += 1
            return b64
        except Exception:
            return ""

    def get_frames(self, lat: float, lon: float) -> Dict:
        """
        Get 6 live camera frames nearest to GPS location.
        Returns dict compatible with nuScenes stream format.
        """
        self._last_lat = lat
        self._last_lon = lon

        # Select 6 nearest cameras
        selected = self.select_nearest(lat, lon, 6)
        self._selected = selected

        # Fetch images (parallel would be faster but keep simple)
        cam_images = []
        cam_names  = []
        cam_dists  = []
        cam_trust  = []

        for i, cam in enumerate(selected):
            b64 = self._fetch_image(cam)
            cam_images.append(b64)
            cam_names.append(cam.name)
            cam_dists.append(round(cam.distance_m/1000, 2))  # km
            # Trust score based on distance (closer = higher trust)
            trust = max(0.5, 1.0 - cam.distance_m/5000)
            cam_trust.append(round(trust, 2))

        # Pad to 6 if needed
        while len(cam_images) < 6:
            cam_images.append("")
            cam_names.append("")
            cam_dists.append(0.0)
            cam_trust.append(0.0)

        return {
            "cam_images":  cam_images,
            "cam_names":   cam_names,
            "cam_dists_km": cam_dists,
            "cam_trust":   cam_trust,
            "cam_roles":   self.CAM_ROLES,
            "n_cameras":   len([x for x in cam_images if x]),
            "nearest_cam": selected[0].name if selected else "",
            "nearest_dist_km": round(selected[0].distance_m/1000,2) if selected else 0,
            "source":      "NYC DOT Live Traffic Cameras",
            "total_nyc_cameras": len(self._cameras),
        }

    def get_camera_info(self) -> List[Dict]:
        """Get info about currently selected cameras."""
        return [
            {
                'name':     cam.name,
                'area':     cam.area,
                'lat':      cam.lat,
                'lon':      cam.lon,
                'dist_km':  round(cam.distance_m/1000, 2),
                'role':     self.CAM_ROLES[i],
            }
            for i, cam in enumerate(self._selected)
        ]


# Singleton
_feed: Optional[NYCLiveCameraFeed] = None

def get_nyc_feed() -> NYCLiveCameraFeed:
    global _feed
    if _feed is None:
        _feed = NYCLiveCameraFeed()
    return _feed
