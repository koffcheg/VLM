import yaml
import numpy as np
import os

def load_zone_config(filepath):
    """
    Loads and parses detection zone configurations from a YAML file.
    Automatically converts RGB color values from the file to BGR for OpenCV.
    Supports range notation for joint_ids (e.g., "0:16").
    """
    if not os.path.exists(filepath):
        print(f"[Mask Logic] Error: Zone configuration file not found at '{filepath}'.")
        return []
        
    try:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
            if not config or 'detection_zones' not in config:
                print(f"[Mask Logic] Warning: YAML file '{filepath}' is empty or missing 'detection_zones' key.")
                return []

            zones = config['detection_zones']
            for zone in zones:
                # --- MODIFICATION: Convert RGB to BGR ---
                if 'color' in zone and len(zone['color']) == 3:
                    # Original color is stored as [R, G, B]
                    r, g, b = zone['color']
                    # Convert to [B, G, R] for OpenCV processing
                    zone['color'] = [b, g, r]
                # --- END MODIFICATION ---

                if 'joint_ids' not in zone:
                    continue
                
                expanded_ids = []
                for item in zone['joint_ids']:
                    if isinstance(item, str) and ':' in item:
                        try:
                            start, end = map(int, item.split(':'))
                            expanded_ids.extend(list(range(start, end + 1)))
                        except ValueError:
                            print(f"[Mask Logic] Warning: Invalid range format '{item}' in zone '{zone['label']}'. Skipping.")
                    elif isinstance(item, int):
                        expanded_ids.append(item)
                
                zone['joint_ids'] = sorted(list(set(expanded_ids)))

            print(f"[Mask Logic] Successfully loaded and parsed {len(zones)} zones from '{filepath}'.")
            return zones

    except (yaml.YAMLError, IOError) as e:
        print(f"[Mask Logic] Error loading or parsing zone configuration file: {e}")
        return []

def check_person_in_zones(kpts, confs, mask_image, zones, pose_conf_threshold, debug_log_handle=None):
    """
    Checks if any specified keypoints of a person are within their designated colored zones.
    Allows for a +/- 1 variance in color values to account for compression artifacts.
    If a debug_log_handle is provided, it writes detailed debug information to the file.
    """
    active_zones_found = []
    if not zones or mask_image is None:
        return active_zones_found

    h, w, _ = mask_image.shape
    
    if debug_log_handle:
        debug_log_handle.write("\n--- Starting Zone Check for New Person ---\n")

    for zone in zones:
        if debug_log_handle:
            # Note: Debug output will show the BGR color now
            debug_log_handle.write(f"  [Debug] Checking Zone: '{zone['label']}' with expected BGR color {zone['color']}\n")
        
        zone_color_bgr = np.array(zone['color'], dtype=np.uint8)
        
        for joint_id in zone['joint_ids']:
            if joint_id >= len(kpts):
                continue
            
            confidence = confs[joint_id]
            if debug_log_handle:
                debug_log_handle.write(f"    [Debug]  -> Joint ID: {joint_id}, Confidence: {confidence:.2f}\n")

            if confidence > pose_conf_threshold:
                point = kpts[joint_id]
                x, y = int(point[0]), int(point[1])
                
                if 0 <= y < h and 0 <= x < w:
                    pixel_color = mask_image[y, x]
                    
                    color_diff = np.abs(pixel_color.astype(np.int16) - zone_color_bgr.astype(np.int16))
                    colors_match = np.all(color_diff <= 1)

                    if debug_log_handle:
                        debug_log_handle.write(f"      - CONFIDENT! Coords: ({x}, {y})\n")
                        debug_log_handle.write(f"      - Mask Color Found: BGR{list(pixel_color)}\n")
                        debug_log_handle.write(f"      - Colors Match (within +/-1 variance)?: {colors_match}\n")

                    if colors_match:
                        if debug_log_handle:
                            debug_log_handle.write(f"      - SUCCESS: Match found for zone '{zone['label']}'!\n")
                        if zone['label'] not in active_zones_found:
                            active_zones_found.append(zone['label'])
                        break
                elif debug_log_handle:
                    debug_log_handle.write(f"      - CONFIDENT but OUT OF BOUNDS at Coords: ({x}, {y})\n")
            elif debug_log_handle:
                 debug_log_handle.write(f"      - Confidence too low. Skipping location check.\n")
        
        if debug_log_handle and zone['label'] in active_zones_found:
             debug_log_handle.write(f"  [Debug] Zone '{zone['label']}' activated. Moving to next zone.\n\n")

    if debug_log_handle:
        debug_log_handle.write(f"--- Zone Check Complete. Active Zones Found: {active_zones_found if active_zones_found else 'None'} ---\n")
        
    return active_zones_found
