import cv2
import torch
import numpy as np
from ultralytics import YOLOE, YOLO, settings
import argparse
import os
import time
import threading
import copy
import json
import math
import yaml
import csv
from datetime import datetime, timezone

# Import the new modular mask logic
from mask_logic import load_zone_config, check_person_in_zones
# Import updated simple MQTT sender with logging and config support
from simple_mqtt_sender import send_pipeline_output, set_log_level

os.environ['ULTRALYTICS_LOGGING_LEVEL'] = 'WARNING'

def load_pytorch_model(model_path, model_type='yoloe', log_level='debug'):
    """Checks if a PyTorch model file exists and loads it."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file '{model_path}' was not found. Please download it first.")
    if log_level != 'silent':
        print(f"Loading PyTorch model from '{model_path}'...")
    model = YOLOE(model_path) if model_type == 'yoloe' else YOLO(model_path)
    if log_level != 'silent':
        print("PyTorch model loaded successfully.")
    return model

def create_save_directories(args):
    """Creates the directories for saving input, output, and JSON files if specified."""
    if args.save_input_path: os.makedirs(args.save_input_path, exist_ok=True)
    if args.save_output_path: os.makedirs(args.save_output_path, exist_ok=True)
    if args.save_json_path: os.makedirs(args.save_json_path, exist_ok=True)
    if args.save_session_path: os.makedirs(args.save_session_path, exist_ok=True)
    if args.log_noise_data: os.makedirs(args.noise_log_path, exist_ok=True)

def calculate_distance(p1, p2):
    """Calculates the Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_angle(p1, p2, p3):
    """Calculates the interior angle at vertex p2 using the dot product."""
    try:
        v1 = (p1[0] - p2[0], p1[1] - p2[1]); v2 = (p3[0] - p2[0], p3[1] - p2[1])
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2); mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        if mag1 == 0 or mag2 == 0: return 0
        cos_angle = max(-1.0, min(1.0, dot_product / (mag1 * mag2)))
        angle_rad = math.acos(cos_angle)
        return math.degrees(angle_rad)
    except Exception:
        return 0

def check_hand_face_pose(kpts, conf, pose_conf_threshold):
    """
    Checks for a specific pose: hands together near the face, arms bent,
    and elbows below the shoulders.
    MODIFIED: Added Nose-to-Wrist and Shoulder Alignment checks.
    """
    # --- Keypoint Indices ---
    NOSE = 0
    L_SHOULDER, R_SHOULDER = 5, 6; L_ELBOW, R_ELBOW = 7, 8; L_WRIST, R_WRIST = 9, 10
    
    # --- Thresholds ---
    PROXIMITY_THRESHOLD = 1.2; MAX_ELEVATION_ABOVE_SHOULDER = 0.5
    MAX_ELEVATION_BELOW_SHOULDER = 0.4; MIN_ARM_ANGLE = 0.0; MAX_ARM_ANGLE = 80.0
    NOSE_WRIST_PROXIMITY_THRESHOLD = 1.1 # New threshold for nose-to-wrist check
    MIN_SHOULDER_ALIGNMENT_RATIO = 0.85 # New threshold to prevent side-on views

    # --- Initialize Detailed Results ---
    checks = {
        "conf_pass": False,
        "shoulder_alignment_pass": False,
        "shoulder_alignment_details": {"value": None, "threshold": MIN_SHOULDER_ALIGNMENT_RATIO},
        "nose_wrist_proximity_pass": False,
        "nose_wrist_details": {"l_dist_ratio": None, "r_dist_ratio": None, "threshold": NOSE_WRIST_PROXIMITY_THRESHOLD},
        "proximity_pass": False,
        "proximity_details": {"value": None, "threshold": PROXIMITY_THRESHOLD},
        "elevation_pass": False,
        "elevation_details": {"value": None, "threshold_upper": MAX_ELEVATION_ABOVE_SHOULDER, "threshold_lower": -MAX_ELEVATION_BELOW_SHOULDER},
        "elbows_below_shoulders_pass": False,
        "l_arm_pass": False,
        "r_arm_pass": False,
        "arm_angle_details": {
            "left_value": None, "right_value": None,
            "threshold_min": MIN_ARM_ANGLE, "threshold_max": MAX_ARM_ANGLE
        },
        "final_result": False
    }

    # --- Confidence Check ---
    required_indices = [NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST]
    if all(conf[i] > pose_conf_threshold for i in required_indices):
        checks["conf_pass"] = True
    else:
        return checks, False # Early exit if keypoints are not confident

    # --- Get Keypoints ---
    nose_pt = kpts[NOSE]
    l_shoulder_pt, r_shoulder_pt = kpts[L_SHOULDER], kpts[R_SHOULDER]
    l_elbow_pt, r_elbow_pt = kpts[L_ELBOW], kpts[R_ELBOW]
    l_wrist_pt, r_wrist_pt = kpts[L_WRIST], kpts[R_WRIST]

    # --- Elbows Below Shoulders Check ---
    if l_elbow_pt[1] > l_shoulder_pt[1] and r_elbow_pt[1] > r_shoulder_pt[1]:
        checks["elbows_below_shoulders_pass"] = True

    # --- Proximity, Elevation, & NEW Checks ---
    shoulder_width = calculate_distance(l_shoulder_pt, r_shoulder_pt)
    if shoulder_width > 0:
        # --- NEW: Shoulder Alignment Check ---
        h_shoulder_dist = abs(l_shoulder_pt[0] - r_shoulder_pt[0])
        shoulder_alignment_ratio = h_shoulder_dist / shoulder_width if shoulder_width > 0 else 0
        checks["shoulder_alignment_details"]["value"] = round(shoulder_alignment_ratio, 3)
        if shoulder_alignment_ratio > MIN_SHOULDER_ALIGNMENT_RATIO:
            checks["shoulder_alignment_pass"] = True

        # --- NEW: Nose-to-Wrist Proximity Check ---
        nose_to_lwrist_dist = calculate_distance(nose_pt, l_wrist_pt)
        nose_to_rwrist_dist = calculate_distance(nose_pt, r_wrist_pt)
        l_dist_ratio = nose_to_lwrist_dist / shoulder_width
        r_dist_ratio = nose_to_rwrist_dist / shoulder_width
        checks["nose_wrist_details"]["l_dist_ratio"] = round(l_dist_ratio, 3)
        checks["nose_wrist_details"]["r_dist_ratio"] = round(r_dist_ratio, 3)
        if l_dist_ratio < NOSE_WRIST_PROXIMITY_THRESHOLD and r_dist_ratio < NOSE_WRIST_PROXIMITY_THRESHOLD:
            checks["nose_wrist_proximity_pass"] = True

        # Proximity
        wrist_dist = calculate_distance(l_wrist_pt, r_wrist_pt)
        wrist_dist_ratio = wrist_dist / shoulder_width
        checks["proximity_details"]["value"] = round(wrist_dist_ratio, 3)
        if wrist_dist_ratio < PROXIMITY_THRESHOLD:
            checks["proximity_pass"] = True

        # Elevation
        shoulder_midpoint = ((l_shoulder_pt[0] + r_shoulder_pt[0]) / 2, (l_shoulder_pt[1] + r_shoulder_pt[1]) / 2)
        wrist_midpoint = ((l_wrist_pt[0] + r_wrist_pt[0]) / 2, (l_wrist_pt[1] + r_wrist_pt[1]) / 2)
        vertical_diff = shoulder_midpoint[1] - wrist_midpoint[1]
        normalized_vertical_diff = vertical_diff / shoulder_width
        checks["elevation_details"]["value"] = round(normalized_vertical_diff, 3)
        if -MAX_ELEVATION_BELOW_SHOULDER < normalized_vertical_diff < MAX_ELEVATION_ABOVE_SHOULDER:
            checks["elevation_pass"] = True

    # --- Arm Angle Check ---
    l_arm_angle = calculate_angle(l_shoulder_pt, l_elbow_pt, l_wrist_pt)
    r_arm_angle = calculate_angle(r_shoulder_pt, r_elbow_pt, r_wrist_pt)
    checks["arm_angle_details"]["left_value"] = round(l_arm_angle, 1)
    checks["arm_angle_details"]["right_value"] = round(r_arm_angle, 1)

    if MIN_ARM_ANGLE < l_arm_angle < MAX_ARM_ANGLE: checks["l_arm_pass"] = True
    if MIN_ARM_ANGLE < r_arm_angle < MAX_ARM_ANGLE: checks["r_arm_pass"] = True

    # --- Final Result ---
    is_pose = all([
        checks["shoulder_alignment_pass"],
        checks["nose_wrist_proximity_pass"],
        checks["proximity_pass"], 
        checks["elevation_pass"], 
        checks["elbows_below_shoulders_pass"], 
        checks["l_arm_pass"], 
        checks["r_arm_pass"]
    ])
    checks["final_result"] = is_pose

    return checks, is_pose

def process_single_frame(frame, pose_model, seg_model, args, mask_image, zone_config, debug_log_handle):
    """
    Runs the full inference pipeline on a single frame and returns structured results.
    """
    if frame is None:
        return None, None

    # --- Initialization ---
    annotated_frame = frame.copy()
    clean_frame_for_crops = frame.copy()
    
    s1_total, s2_total = 0, 0
    num_people, ignored_person_count = 0, 0
    seg_found = False
    detected_people_list = []

    is_masking_active = mask_image is not None and zone_config is not None

    # --- Stage 1: Pose Detection ---
    if not args.disable_stage_one and pose_model:
        pose_results = pose_model.predict(annotated_frame, verbose=False, classes=[0], conf=args.person_conf)
        s1_pre, s1_inf, s1_post = pose_results[0].speed.values()
        s1_total = s1_pre + s1_inf + s1_post

        person_indices_to_process = list(range(len(pose_results[0].boxes)))
        pose_indices_to_ignore = set()
        
        if is_masking_active and len(person_indices_to_process) > 0:
            indices_to_keep = []
            for i in person_indices_to_process:
                kpts = pose_results[0].keypoints.xy[i].cpu().numpy()
                conf = pose_results[0].keypoints.conf[i].cpu().numpy()
                debug_handle_for_check = debug_log_handle if args.log_level == 'debug' else None
                active_zones = check_person_in_zones(kpts, conf, mask_image, zone_config, args.pose_conf, debug_log_handle=debug_handle_for_check)
                
                if active_zones:
                    indices_to_keep.append((i, active_zones))
                else:
                    ignored_person_count += 1
                    if args.ignore_pose_only:
                        indices_to_keep.append((i, [])) # Keep person but with no zones
                        pose_indices_to_ignore.add(i)
            
            person_indices_to_process = indices_to_keep
        else: # If not masking, prepare all people for processing
            person_indices_to_process = [(i, []) for i in person_indices_to_process]
        
        num_people = len(person_indices_to_process)
        annotated_frame = pose_results[0].plot()

        if num_people > 0 and getattr(pose_results[0], 'keypoints', None) is not None:
            for person_index, active_zones in person_indices_to_process:
                if person_index in pose_indices_to_ignore:
                    continue

                person_kpts = pose_results[0].keypoints.xy[person_index].cpu().numpy()
                person_conf = pose_results[0].keypoints.conf[person_index].cpu().numpy()
                pose_check_details, is_pose_detected = check_hand_face_pose(person_kpts, person_conf, args.pose_conf)

                person_data = {
                    "person_index": person_index,
                    "active_zones": active_zones,
                    "pose_analysis": {
                        "is_target_pose_raw": is_pose_detected,
                        "check_details": {k: bool(v) if isinstance(v, np.bool_) else v for k, v in pose_check_details.items()}
                    }
                }
                detected_people_list.append(person_data)

    # --- Stage 2: Segmentation ---
    if not args.disable_stage_two and seg_model and num_people > 0:
        s2_pre_total, s2_inf_total, s2_post_total = 0,0,0
        for person_index, _ in person_indices_to_process:
            box = pose_results[0].boxes[person_index]
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            person_crop = clean_frame_for_crops[y1:y2, x1:x2]
            if person_crop.size > 0:
                seg_results = seg_model.predict(person_crop, verbose=False)
                s2_pre, s2_inf, s2_post = seg_results[0].speed.values()
                s2_pre_total += s2_pre; s2_inf_total += s2_inf; s2_post_total += s2_post
                if seg_results[0].masks: 
                    seg_found = True
                    annotated_frame[y1:y2, x1:x2] = seg_results[0].plot()
        s2_total = s2_pre_total + s2_inf_total + s2_post_total
    
    # --- Assemble Frame Results ---
    frame_results = {
        "performance_ms": {
            "stage1_pose": round(s1_total, 1),
            "stage2_segmentation": round(s2_total, 1)
        },
        "detection_summary": {
            "people_detected": num_people,
            "people_ignored_by_mask": ignored_person_count
        },
        "detected_people": detected_people_list
    }
    
    return frame_results, annotated_frame

# --- Global Variables for Frame Buffers and Thread Control ---
latest_frame = None; latest_result_frame = None; lock = threading.Lock(); stop_event = threading.Event()
latest_frame_2 = None; lock_2 = threading.Lock()
# --- NEW: Globals for dedicated save streams ---
latest_save_frame_1 = None; lock_save_1 = threading.Lock()
latest_save_frame_2 = None; lock_save_2 = threading.Lock()


def camera_thread_func(cap):
    global latest_frame, lock, stop_event
    print("[Camera Thread 1] Started."); time.sleep(2); print("[Camera Thread 1] Starting frame capture loop.")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret: print("[Camera Thread 1] Stream ended or failed."); stop_event.set(); break
        with lock: latest_frame = frame
        time.sleep(0.01)
    print("[Camera Thread 1] Stopped.")

def camera_thread_2_func(cap):
    global latest_frame_2, lock_2, stop_event
    print("[Camera Thread 2] Started."); time.sleep(2); print("[Camera Thread 2] Starting frame capture loop.")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret: print("[Camera Thread 2] Stream ended or failed."); break
        with lock_2: latest_frame_2 = frame
        time.sleep(0.01)
    print("[Camera Thread 2] Stopped.")

# --- NEW: Camera thread functions for the dedicated save streams ---
def save_camera_thread_1_func(cap):
    global latest_save_frame_1, lock_save_1, stop_event
    print("[Save Stream 1] Started."); time.sleep(2); print("[Save Stream 1] Starting frame capture loop.")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret: print("[Save Stream 1] Stream ended or failed."); break
        with lock_save_1: latest_save_frame_1 = frame
        time.sleep(0.01)
    print("[Save Stream 1] Stopped.")

def save_camera_thread_2_func(cap):
    global latest_save_frame_2, lock_save_2, stop_event
    print("[Save Stream 2] Started."); time.sleep(2); print("[Save Stream 2] Starting frame capture loop.")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret: print("[Save Stream 2] Stream ended or failed."); break
        with lock_save_2: latest_save_frame_2 = frame
        time.sleep(0.01)
    print("[Save Stream 2] Stopped.")


def combine_frames(primary_frame, secondary_frame):
    """Resizes secondary frame to match primary's width and combines them vertically."""
    if primary_frame is None and secondary_frame is None: return None
    if primary_frame is None: return secondary_frame
    if secondary_frame is None: return primary_frame

    h1, w1, _ = primary_frame.shape
    h2, w2, _ = secondary_frame.shape
    if w1 == 0 or w2 == 0: return primary_frame # Avoid division by zero
    new_h2 = int(h2 * (w1 / w2))
    resized_secondary = cv2.resize(secondary_frame, (w1, new_h2))
    return cv2.vconcat([primary_frame, resized_secondary])

def display_processing_thread_func(pose_model, seg_model, args, mask_image=None, zone_config=None, debug_log_handle=None):
    global latest_frame, latest_result_frame, lock, stop_event, latest_frame_2, lock_2
    global latest_save_frame_1, lock_save_1, latest_save_frame_2, lock_save_2 # Access new globals
    print("[Processing Thread] Started.")

    last_save_time = 0; save_index = 0; session_active = False; session_id = None
    current_session_path = None; last_person_detected_time = 0
    session_counter = 0; csv_writer = None; csv_file = None
    
    pose_streak_start_time = None
    pose_confirmed_count = 0
    was_pose_confirmed_in_streak = False
    
    if args.log_noise_data:
        # CSV Logging setup remains the same
        pass

    while not stop_event.is_set():
        # --- Frame Acquisition ---
        with lock: frame_1 = copy.deepcopy(latest_frame)
        frame_2 = None
        if args.video_source_2:
            with lock_2: frame_2 = copy.deepcopy(latest_frame_2)
        
        # --- NEW: Acquire High-Quality Save Frames (if available) ---
        save_frame_1 = None
        if args.save_stream_url_1:
            with lock_save_1:
                save_frame_1 = copy.deepcopy(latest_save_frame_1)
        
        save_frame_2 = None
        if args.save_stream_url_2:
            with lock_save_2:
                save_frame_2 = copy.deepcopy(latest_save_frame_2)

        if frame_1 is None:
            time.sleep(0.1)
            continue
        
        # --- Main Processing ---
        start_time = time.monotonic()
        results_1, annotated_frame_1 = process_single_frame(frame_1, pose_model, seg_model, args, mask_image, zone_config, debug_log_handle)
        
        results_2, annotated_frame_2 = None, None
        if args.process_both_streams and frame_2 is not None:
            results_2, annotated_frame_2 = process_single_frame(frame_2, pose_model, seg_model, args, mask_image, zone_config, debug_log_handle)

        # --- Session Management (MOVED BEFORE MQTT SEND) ---
        total_people_detected = 0
        if results_1: total_people_detected += results_1['detection_summary']['people_detected']
        if results_2: total_people_detected += results_2['detection_summary']['people_detected']
        
        current_time = time.monotonic()
        if args.save_session_path:
            if total_people_detected > 0:
                last_person_detected_time = current_time
                if not session_active:
                    session_active = True; session_counter += 1; session_id = time.strftime("%Y%m%d-%H%M%S")
                    date_folder, time_folder = time.strftime("%Y-%m-%d"), time.strftime("%H-%M-%S")
                    current_session_path = os.path.join(args.save_session_path, date_folder, time_folder)
                    os.makedirs(current_session_path, exist_ok=True)
                    if args.log_level == 'debug': print(f"--- New Session Started (ID: {session_id}) ---")
            elif session_active and (current_time - last_person_detected_time > args.session_timeout):
                if args.log_level == 'debug': print(f"--- Session Ended (ID: {session_id}) ---")
                session_active = False; session_id = None; current_session_path = None

        # --- Data Aggregation and JSON Creation (WITH UPDATED SESSION STATE) ---
        is_masking_enabled = mask_image is not None and zone_config is not None
        pipeline_output = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "masking_enabled": is_masking_enabled,
            "session_info": {"is_active": session_active, "session_id": session_id},
            "pipeline_sources": {
                "primary": results_1 if results_1 else {},
                "secondary": results_2 if results_2 else {}
            }
        }

        # --- Pose Status Calculation (BEFORE MQTT send) ---
        is_raw_pose_detected_this_frame = False
        primary_results = pipeline_output.get("pipeline_sources", {}).get("primary", {})
        if primary_results and primary_results.get("detected_people"):
            for person in primary_results["detected_people"]:
                if person.get("pose_analysis", {}).get("is_target_pose_raw"):
                    is_raw_pose_detected_this_frame = True
                    break
        
        is_pose_confirmed_this_frame = False
        if is_raw_pose_detected_this_frame:
            if pose_streak_start_time is None: 
                pose_streak_start_time = time.monotonic()
            if time.monotonic() - pose_streak_start_time >= args.pose_hold_time:
                is_pose_confirmed_this_frame = True
        else:
            pose_streak_start_time = None
        
        if is_pose_confirmed_this_frame and not was_pose_confirmed_in_streak:
            pose_confirmed_count += 1
        
        if is_pose_confirmed_this_frame:
            was_pose_confirmed_in_streak = True
        elif not is_raw_pose_detected_this_frame:
            was_pose_confirmed_in_streak = False

        # Add is_pose_confirmed field to pipeline_output BEFORE sending MQTT
        if primary_results and primary_results.get("detected_people"):
            for person in primary_results["detected_people"]:
                if "pose_analysis" in person:
                    person["pose_analysis"]["is_pose_confirmed"] = False
            
            if is_pose_confirmed_this_frame:
                for person in primary_results["detected_people"]:
                    if person.get("pose_analysis", {}).get("is_target_pose_raw"):
                        person["pose_analysis"]["is_pose_confirmed"] = True
                        break

        # Send MQTT messages with complete pipeline data (including is_pose_confirmed)
        send_pipeline_output(pipeline_output)

        if debug_log_handle:
            debug_log_handle.write("\n--- FRAME JSON OUTPUT ---\n")
            json.dump(pipeline_output, debug_log_handle, indent=4)
            debug_log_handle.write("\n--- END FRAME JSON OUTPUT ---\n\n")
            debug_log_handle.flush()

        # --- Display and Logging ---
        total_pipeline_time_ms = (current_time - start_time) * 1000
        perf_text = [
            f"Pipeline Time: {total_pipeline_time_ms:.1f}ms",
            f"Detections (P/S): {results_1['detection_summary']['people_detected'] if results_1 else 'N/A'} / {results_2['detection_summary']['people_detected'] if results_2 else 'N/A'}",
            f"Ignored (P/S): {results_1['detection_summary']['people_ignored_by_mask'] if results_1 else 'N/A'} / {results_2['detection_summary']['people_ignored_by_mask'] if results_2 else 'N/A'}",
            f"Sessions Detected: {session_counter}",
            f"Session: {session_id if session_active else 'INACTIVE'}",
            f"Pose Confirmed Count: {pose_confirmed_count}"
        ]

        pose_text_lines = []
        pose_check_results_for_display = None
        if primary_results and primary_results.get("detected_people"):
            if primary_results["detected_people"]:
                first_person = primary_results["detected_people"][0]
                pose_analysis = first_person.get("pose_analysis", {})
                if pose_analysis:
                    pose_check_results_for_display = pose_analysis.get("check_details")

        # --- MODIFICATION: Create detailed pose text for both video overlay and terminal ---
        if pose_check_results_for_display:
            pr = pose_check_results_for_display
            status_color, fail_color = (0, 255, 0), (0, 0, 255)
            pose_text_lines.append(("-- Pose Check (Primary) --", status_color))

            conf_pass = pr.get('conf_pass')
            pose_text_lines.append((f"Keypoint Conf: {'PASS' if conf_pass else 'FAIL'}", status_color if conf_pass else fail_color))

            # NEW: Shoulder Alignment Display
            shoulder_pass = pr.get('shoulder_alignment_pass')
            shoulder_details = pr.get('shoulder_alignment_details', {})
            shoulder_val, shoulder_thresh = shoulder_details.get('value'), shoulder_details.get('threshold')
            shoulder_text = f"Shoulder Align: {'PASS' if shoulder_pass else 'FAIL'}"
            if shoulder_val is not None: shoulder_text += f" ({shoulder_val:.2f} > {shoulder_thresh})"
            pose_text_lines.append((shoulder_text, status_color if shoulder_pass else fail_color))

            # NEW: Nose to Wrist Proximity Display
            nose_pass = pr.get('nose_wrist_proximity_pass')
            nose_details = pr.get('nose_wrist_details', {})
            l_dist, r_dist, nose_thresh = nose_details.get('l_dist_ratio'), nose_details.get('r_dist_ratio'), nose_details.get('threshold')
            nose_text = f"Nose-Wrist Prox: {'PASS' if nose_pass else 'FAIL'}"
            if l_dist is not None and r_dist is not None: nose_text += f" (L:{l_dist:.2f}, R:{r_dist:.2f} < {nose_thresh})"
            pose_text_lines.append((nose_text, status_color if nose_pass else fail_color))
            
            # Wrist Proximity Display
            prox_pass = pr.get('proximity_pass')
            prox_details = pr.get('proximity_details', {})
            prox_val, prox_thresh = prox_details.get('value'), prox_details.get('threshold')
            prox_text = f"Wrist Proximity: {'PASS' if prox_pass else 'FAIL'}"
            if prox_val is not None: prox_text += f" ({prox_val:.2f} < {prox_thresh})"
            pose_text_lines.append((prox_text, status_color if prox_pass else fail_color))
            
            # Wrist Elevation Display
            elev_pass = pr.get('elevation_pass')
            elev_details = pr.get('elevation_details', {})
            elev_val = elev_details.get('value')
            elev_thresh_u, elev_thresh_l = elev_details.get('threshold_upper'), elev_details.get('threshold_lower')
            elev_text = f"Wrist Elevation: {'PASS' if elev_pass else 'FAIL'}"
            if elev_val is not None: elev_text += f" ({elev_thresh_l:.1f} < {elev_val:.2f} < {elev_thresh_u:.1f})"
            pose_text_lines.append((elev_text, status_color if elev_pass else fail_color))

            # Elbows Below Shoulders Display
            elbow_pass = pr.get('elbows_below_shoulders_pass')
            pose_text_lines.append((f"Elbows Below Shoulders: {'PASS' if elbow_pass else 'FAIL'}", status_color if elbow_pass else fail_color))

            # Arm Angles Display
            l_arm_pass, r_arm_pass = pr.get('l_arm_pass'), pr.get('r_arm_pass')
            arm_details = pr.get('arm_angle_details', {})
            l_val, r_val = arm_details.get('left_value'), arm_details.get('right_value')
            arm_text = f"Arm Angles: {'PASS' if l_arm_pass and r_arm_pass else 'FAIL'}"
            if l_val is not None and r_val is not None: arm_text += f" (L:{l_val:.1f}, R:{r_val:.1f})"
            pose_text_lines.append((arm_text, status_color if l_arm_pass and r_arm_pass else fail_color))
            
            if is_pose_confirmed_this_frame:
                final_status, final_color = "POSE CONFIRMED", status_color
            elif is_raw_pose_detected_this_frame:
                final_status, final_color = "HOLDING...", (255, 165, 0) # Orange
            else:
                final_status, final_color = "POSE NOT DETECTED", fail_color
            pose_text_lines.append((final_status, final_color))

        display_frame = annotated_frame_1 if annotated_frame_1 is not None else frame_1
        if args.display_video and display_frame is not None:
            total_lines = len(perf_text) + len(pose_text_lines)
            box_height = 25 + total_lines * 23 + (10 if pose_text_lines else 0)
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (5, 5), (460, box_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, dst=display_frame)
            
            y_pos = 25
            for line in perf_text:
                cv2.putText(display_frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y_pos += 23
            
            if pose_text_lines:
                y_pos += 10
                for line, color in pose_text_lines:
                    cv2.putText(display_frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_pos += 23
        
        # --- MODIFICATION: Print detailed pose info to terminal in debug mode ---
        if args.log_level == 'debug':
            print("\033[H\033[J", end="")
            [print(line) for line in perf_text]
            print("-" * 25)
            if pose_text_lines:
                [print(line[0]) for line in pose_text_lines]
                print("-" * 25)

        # --- Saving Logic ---
        if current_time - last_save_time >= args.save_interval:
            save_index_str = str(save_index)
            if args.max_saves > 0: save_index = (save_index + 1) % args.max_saves
            
            if args.save_json_path:
                json_filename = os.path.join(args.save_json_path, f"{save_index_str}_data.json")
                with open(json_filename, 'w') as f:
                    json.dump(pipeline_output, f, indent=4)
            
            # --- MODIFIED SAVING LOGIC ---
            if args.save_output_path:
                # Prioritize the dedicated save stream frame (un-annotated), otherwise use the annotated frame.
                output_frame_to_save_1 = save_frame_1 if save_frame_1 is not None else annotated_frame_1
                output_frame_to_save_2 = save_frame_2 if save_frame_2 is not None else annotated_frame_2

                if output_frame_to_save_1 is not None:
                    cv2.imwrite(os.path.join(args.save_output_path, f"{save_index_str}_output_primary.jpg"), output_frame_to_save_1)
                if output_frame_to_save_2 is not None:
                    cv2.imwrite(os.path.join(args.save_output_path, f"{save_index_str}_output_secondary.jpg"), output_frame_to_save_2)

            if args.save_input_path:
                # Prioritize the dedicated save stream frame, otherwise use the raw processing frame.
                input_frame_to_save_1 = save_frame_1 if save_frame_1 is not None else frame_1
                input_frame_to_save_2 = save_frame_2 if save_frame_2 is not None else frame_2
                
                if input_frame_to_save_1 is not None:
                    cv2.imwrite(os.path.join(args.save_input_path, f"{save_index_str}_input_primary.jpg"), input_frame_to_save_1)
                if input_frame_to_save_2 is not None:
                    cv2.imwrite(os.path.join(args.save_input_path, f"{save_index_str}_input_secondary.jpg"), input_frame_to_save_2)
            # --- END MODIFIED SAVING LOGIC ---

            if session_active and current_session_path:
                session_img_filename = f"img_{int(current_time * 1e6)}"
                
                primary_frame_to_save = frame_1 if args.session_save_raw else annotated_frame_1
                
                secondary_frame_to_save = None
                if args.video_source_2:
                    # If saving raw, always use the raw frame.
                    if args.session_save_raw:
                        secondary_frame_to_save = frame_2
                    # Otherwise, use the annotated frame if available.
                    elif args.process_both_streams:
                        secondary_frame_to_save = annotated_frame_2
                    # Fallback to the raw frame if not processing the stream.
                    else:
                        secondary_frame_to_save = frame_2
                
                if args.combine_video_sources:
                    combined_output = combine_frames(primary_frame_to_save, secondary_frame_to_save)
                    if combined_output is not None:
                        cv2.imwrite(os.path.join(current_session_path, f"{session_img_filename}_combined.jpg"), combined_output)
                else:
                    if primary_frame_to_save is not None:
                        cv2.imwrite(os.path.join(current_session_path, f"{session_img_filename}_primary.jpg"), primary_frame_to_save)
                    if secondary_frame_to_save is not None:
                        cv2.imwrite(os.path.join(current_session_path, f"{session_img_filename}_secondary.jpg"), secondary_frame_to_save)
            
            last_save_time = current_time

        with lock: latest_result_frame = display_frame

    if csv_file: print("[Noise Logger] Closing noise log file."); csv_file.close()
    print("[Processing Thread] Stopped.")

def main(args):
    pose_model, seg_model = None, None
    debug_log_handle = None 

    # Set MQTT logging level to match run_inference log level
    set_log_level(args.log_level)

    if args.log_level == 'debug' and args.debug_log_file:
        try:
            debug_log_handle = open(args.debug_log_file, 'w')
            print(f"[Debug] Logging detailed checks to {args.debug_log_file}")
        except IOError as e:
            print(f"Warning: Could not open debug log file {args.debug_log_file} for writing. {e}")
            debug_log_handle = None

    if not args.disable_stage_one: pose_model = load_pytorch_model("ecop_v3.pt", model_type='yolo', log_level=args.log_level)
    if not args.disable_stage_two:
        seg_model = load_pytorch_model("ecoes_v3.pt", model_type='yoloe', log_level=args.log_level)
        names = [args.prompt] if args.detection_mode == 'direct_prompt' else ["t-shirt"]
        if args.log_level != 'silent': print(f"Setting Stage 2 detection class(es) to: {names}")
        seg_model.set_classes(names, seg_model.get_text_pe(names))

    create_save_directories(args)
    
    mask_image = None
    zone_config = None 
    if args.input_mask:
        if not os.path.exists(args.input_mask):
            print(f"Warning: The mask file '{args.input_mask}' was not found. Continuing without a mask.")
            args.input_mask = None
        else:
            mask_image = cv2.imread(args.input_mask)
            if mask_image is None:
                print(f"Warning: Failed to load mask image from '{args.input_mask}'.")
            else:
                print(f"Successfully loaded input mask from '{args.input_mask}'.")
                if args.zone_config:
                    zone_config = load_zone_config(args.zone_config)
                    if not zone_config:
                        print("Warning: Could not load zone configuration. Zone checking disabled.")
                else:
                    print("Warning: `input_mask` is specified but `zone_config` is missing. Zone checking disabled.")
                    zone_config = None
    
    if args.image_source: print("Image source mode not supported."); return
    elif args.video_source:
        def create_gstreamer_pipeline(uri, latency=0): return (f"rtspsrc location={uri} latency={latency} ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink max-buffers=1 drop=true")

        # --- Setup Main Inference Streams ---
        if str(args.video_source).startswith('rtsp://'):
            print(f"Opening GStreamer pipeline for RTSP source: {args.video_source}")
            pipeline_1 = create_gstreamer_pipeline(args.video_source)
            cap_1 = cv2.VideoCapture(pipeline_1, cv2.CAP_GSTREAMER)
        else:
            print(f"Opening local video file: {args.video_source}")
            cap_1 = cv2.VideoCapture(args.video_source)
            
        if not cap_1.isOpened():
            print("Error: Failed to open primary video stream. Exiting.")
            return
        
        threads = []; cam_thread_1 = threading.Thread(target=camera_thread_func, args=(cap_1,)); threads.append(cam_thread_1)
        cap_2 = None
        
        if args.video_source_2:
            if str(args.video_source_2).startswith('rtsp://'):
                print(f"Opening GStreamer pipeline for secondary RTSP source: {args.video_source_2}")
                pipeline_2 = create_gstreamer_pipeline(args.video_source_2)
                cap_2 = cv2.VideoCapture(pipeline_2, cv2.CAP_GSTREAMER)
            else:
                print(f"Opening local secondary video file: {args.video_source_2}")
                cap_2 = cv2.VideoCapture(args.video_source_2)
                
            if cap_2.isOpened():
                cam_thread_2 = threading.Thread(target=camera_thread_2_func, args=(cap_2,)); threads.append(cam_thread_2)
            else:
                print("Warning: Failed to open secondary video stream. Continuing without it.")
                args.video_source_2 = None
        
        # --- NEW: Setup Dedicated Save Streams ---
        cap_save_1, cap_save_2 = None, None
        if args.save_stream_url_1:
            print(f"Opening GStreamer pipeline for primary SAVE stream: {args.save_stream_url_1}")
            pipeline_save_1 = create_gstreamer_pipeline(args.save_stream_url_1)
            cap_save_1 = cv2.VideoCapture(pipeline_save_1, cv2.CAP_GSTREAMER)
            if cap_save_1.isOpened():
                save_thread_1 = threading.Thread(target=save_camera_thread_1_func, args=(cap_save_1,)); threads.append(save_thread_1)
            else:
                print("Warning: Failed to open primary save stream. It will not be used.")
                args.save_stream_url_1 = None
        
        if args.save_stream_url_2:
            print(f"Opening GStreamer pipeline for secondary SAVE stream: {args.save_stream_url_2}")
            pipeline_save_2 = create_gstreamer_pipeline(args.save_stream_url_2)
            cap_save_2 = cv2.VideoCapture(pipeline_save_2, cv2.CAP_GSTREAMER)
            if cap_save_2.isOpened():
                save_thread_2 = threading.Thread(target=save_camera_thread_2_func, args=(cap_save_2,)); threads.append(save_thread_2)
            else:
                print("Warning: Failed to open secondary save stream. It will not be used.")
                args.save_stream_url_2 = None

        proc_thread = threading.Thread(target=display_processing_thread_func, args=(pose_model, seg_model, args, mask_image, zone_config, debug_log_handle))
        threads.append(proc_thread)

        for t in threads: t.start()
        try:
            if args.display_video:
                while not stop_event.is_set():
                    with lock: display_image = copy.deepcopy(latest_result_frame)
                    if display_image is not None: cv2.imshow('Live Inference Pipeline', display_image)
                    if cv2.waitKey(30) & 0xFF == ord('q'): stop_event.set()
                cv2.destroyAllWindows()
            else:
                while not stop_event.is_set(): time.sleep(1)
        except KeyboardInterrupt: print("\nShutdown signal (Ctrl+C) received. Stopping threads..."); stop_event.set()
        
        for t in threads: t.join()
        
        # --- Release all VideoCapture objects ---
        cap_1.release()
        if cap_2: cap_2.release()
        if cap_save_1: cap_save_1.release()
        if cap_save_2: cap_save_2.release()
        
        if debug_log_handle:
            debug_log_handle.close()
            print("[Debug] Debug log file closed.")

        print("Pipeline stopped gracefully.")

if __name__ == '__main__':
    conf_parser = argparse.ArgumentParser(description="Configuration loader for the AI pipeline.", add_help=False)
    conf_parser.add_argument("--config", default="config.yaml", help="Path to the YAML configuration file. Defaults to config.yaml")
    conf_args, remaining_argv = conf_parser.parse_known_args()

    yaml_defaults = {}
    if conf_args.config and os.path.exists(conf_args.config):
        print(f"Loading configuration from: {conf_args.config}")
        with open(conf_args.config, 'r') as f:
            try: yaml_defaults = yaml.safe_load(f) or {}
            except yaml.YAMLError as exc: print(f"Error parsing YAML file: {exc}")

    parser = argparse.ArgumentParser(parents=[conf_parser], description="A multi-modal AI inference pipeline with real-time pose detection and session management.")
    
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument('--video-source', type=str, help='Primary input video source for live mode (e.g., rtsp://...).')
    source_group.add_argument('--image-source', type=str, help='Path to a single image (NOTE: live display not implemented for image mode).')
    
    parser.add_argument('--log-level', type=str, choices=['debug', 'silent'], help='Set terminal logging level.')
    parser.add_argument('--debug-log-file', type=str, help="Path to a file to store detailed debug logs. Only active when --log-level is 'debug'.")
    parser.add_argument('--video-source-2', type=str, help='Optional second video source for saving frames only.')
    
    # --- NEW: Arguments for dedicated save streams ---
    parser.add_argument('--save-stream-url-1', type=str, help='Optional high-res RTSP stream for saving live images from the primary source.')
    parser.add_argument('--save-stream-url-2', type=str, help='Optional high-res RTSP stream for saving live images from the secondary source.')
    
    parser.add_argument('--combine-video-sources', action=argparse.BooleanOptionalAction, help="Combine frames from both sources into one image.")
    parser.add_argument('--display-video', action='store_true', help='Enable a live display window.')
    
    parser.add_argument('--save-input-path', type=str, help='Path for generic (non-session) raw input frames.')
    parser.add_argument('--save-output-path', type=str, help='Path for generic (non-session) final annotated frames.')
    parser.add_argument('--save-json-path', type=str, help='Path for generic (non-session) detection and keypoint data.')
    parser.add_argument('--save-session-path', type=str, help='Root path to save full, non-overwriting session recordings.')
    parser.add_argument('--session-save-raw', action='store_true', help='If set, saves raw input frames during sessions instead of annotated frames.')
    parser.add_argument('--session-timeout', type=float, default=40.0, help='Time in seconds to wait before ending a session.')
    parser.add_argument('--save-interval', type=float, help='Minimum time interval in seconds between saves.')
    parser.add_argument('--max-saves', type=int, help='For generic saves, sets max files to keep (circular buffer).')
    parser.add_argument('--log-noise-data', action='store_true', help='Enable logging of session/pose detection stats to a CSV file for noise analysis.')
    parser.add_argument('--noise-log-path', type=str, default='.', help='Path to save the noise data CSV file.')
    parser.add_argument('--disable-stage-one', action='store_true', help='Disable the first stage (pose model).')
    parser.add_argument('--disable-stage-two', action='store_true', help='Disable the second stage (segmentation model).')
    parser.add_argument('--process-both-streams', action=argparse.BooleanOptionalAction, help='Run inference on both primary and secondary video streams.')
    parser.add_argument('--person-conf', type=float, help='Confidence threshold for person detection (0.0 to 1.0).')
    parser.add_argument('--pose-conf', type=float, help='Confidence threshold for individual pose keypoints (0.0 to 1.0).')
    parser.add_argument('--pose-hold-time', type=float, help='Time in seconds the pose must be held continuously to be confirmed.')
    parser.add_argument('--input-mask', type=str, help='Path to an input mask image. Detections will be filtered based on this mask.')
    parser.add_argument('--zone-config', type=str, default='detection_zones.yaml', help='Path to the YAML file that defines the meaning of the colors in the mask image.')
    parser.add_argument('--ignore-pose-only', action='store_true', help='If true, only ignores the pose detection for people outside a valid zone, not the person detection itself.')
    parser.add_argument('--detection-mode', type=str, choices=['direct_prompt', 'color_analysis'], help="Mode for the segmentation model.")
    parser.add_argument('--prompt', type=str, help="The text prompt for segmentation detection.")

    parser.set_defaults(**yaml_defaults)
    args = parser.parse_args(remaining_argv)
    
    main(args)