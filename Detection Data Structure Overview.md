## **Detection Data Structure Overview**

The `pipeline_output` object is the main container for all data generated during a single frame's inference cycle. It includes timestamps, session information, masking status, and detailed results from one or two video streams.

JSON  
{  
    "timestamp": "2025-07-31T16:05:32.123456+00:00",  
    "masking\_enabled": true,  
    "session\_info": {  
        "is\_active": true,  
        "session\_id": "20250731-160530"  
    },  
    "pipeline\_sources": {  
        "primary": {  
            "performance\_ms": {  
                "stage1\_pose": 25.5,  
                "stage2\_segmentation": 15.2  
            },  
            "detection\_summary": {  
                "people\_detected": 2,  
                "people\_ignored\_by\_mask": 1  
            },  
            "detected\_people": \[  
                {  
                    "person\_index": 0,  
                    "active\_zones": \[  
                        "zone\_A",  
                        "Customer Interaction Zone"  
                    \],  
                    "pose\_analysis": {  
                        "is\_target\_pose\_raw": true,  
                        "is\_pose\_confirmed": true,  
                        "check\_details": {  
                            "conf\_pass": true,  
                            "proximity\_pass": true,  
                            "proximity\_details": {  
                                "value": 0.95,  
                                "threshold": 1.2  
                            },  
                            "elevation\_pass": true,  
                            "elevation\_details": {  
                                "value": 0.25,  
                                "threshold\_upper": 0.5,  
                                "threshold\_lower": \-0.4  
                            },  
                            "elbows\_below\_shoulders\_pass": true,  
                            "l\_arm\_pass": true,  
                            "r\_arm\_pass": true,  
                            "arm\_angle\_details": {  
                                "left\_value": 45.5,  
                                "right\_value": 50.1,  
                                "threshold\_min": 0.0,  
                                "threshold\_max": 80.0  
                            },  
                            "final\_result": true  
                        }  
                    }  
                },  
                {  
                    "person\_index": 1,  
                    "active\_zones": \[  
                        "zone\_A",  
                        "zone\_B"  
                    \],  
                    "pose\_analysis": {  
                        "is\_target\_pose\_raw": false,  
                        "is\_pose\_confirmed": false,  
                        "check\_details": {  
                            "conf\_pass": true,  
                            "proximity\_pass": false,  
                            "proximity\_details": {  
                                "value": 1.51,  
                                "threshold": 1.2  
                            },  
                            "elevation\_pass": true,  
                            "elevation\_details": {  
                                "value": 0.1,  
                                "threshold\_upper": 0.5,  
                                "threshold\_lower": \-0.4  
                            },  
                            "elbows\_below\_shoulders\_pass": true,  
                            "l\_arm\_pass": true,  
                            "r\_arm\_pass": true,  
                            "arm\_angle\_details": {  
                                "left\_value": 65,  
                                "right\_value": 70.2,  
                                "threshold\_min": 0.0,  
                                "threshold\_max": 80.0  
                            },  
                            "final\_result": false  
                        }  
                    }  
                }  
            \]  
        },  
        "secondary": {}  
    }  
}

---

## **Field Documentation**

### **Top-Level Fields**

* **`timestamp`** (string): The UTC timestamp in ISO 8601 format indicating when the frame was processed.  
* **`masking_enabled`** (boolean): `true` if a detection mask is active; otherwise `false`.  
* **`session_info`** (object): Contains information about the current detection session.  
  * **`is_active`** (boolean): `true` if a person has been detected recently and a session is considered active.  
  * **`session_id`** (string | null): A unique identifier for the active session (e.g., `YYYYMMDD-HHMMSS`). It is `null` if no session is active.  
* **`pipeline_sources`** (object): Contains the inference results for each video stream.  
  * **`primary`** (object): Results from the main video source. Can be an empty object if no frame was processed.  
  * **`secondary`** (object): Results from the optional second video source. Empty if not in use.

### **`primary` / `secondary` Object Fields**

* **`performance_ms`** (object): Timing information for the inference stages in milliseconds.  
  * **`stage1_pose`** (float): Total time for the pose detection model.  
  * **`stage2_segmentation`** (float): Total time for the segmentation model.  
* **`detection_summary`** (object): A summary of detections in the frame.  
  * **`people_detected`** (integer): The number of people detected and processed (after filtering by the mask).  
  * **`people_ignored_by_mask`** (integer): The number of people detected but subsequently ignored because they were outside a valid mask zone.  
* **`detected_people`** (array): A list of objects, where each object contains details for one detected person.

### **`detected_people` Array Object Fields**

* **`person_index`** (integer): The internal index of the person as identified by the pose model for that frame.  
* **`active_zones`** (array of strings): A list of zone names (from `detection_zones.yaml`) that the person's keypoints occupy. Empty if no zones are occupied or if masking is disabled.  
* **`pose_analysis`** (object): Contains the results of the specific pose check.  
  * **`is_target_pose_raw`** (boolean): `true` if the person's pose matches the target criteria in this specific frame.  
  * **`is_pose_confirmed`** (boolean): `true` if the target pose has been held continuously for the duration specified by `--pose-hold-time`.  
  * **`check_details`** (object): A detailed breakdown of each check performed to determine the pose.  
    * **`conf_pass`** (boolean): `true` if all required keypoints have a confidence score above the `--pose-conf` threshold.  
    * **`proximity_pass`** (boolean): `true` if the wrist proximity check passed.  
    * **`proximity_details`** (object): Contains the calculated value and threshold for the proximity check.  
    * **`elevation_pass`** (boolean): `true` if the wrist elevation check passed.  
    * **`elevation_details`** (object): Contains the value and thresholds for the elevation check.  
    * **`elbows_below_shoulders_pass`** (boolean): `true` if both elbows are below the shoulders.  
    * **`l_arm_pass`** / **`r_arm_pass`** (boolean): `true` if the respective arm angle check passed.  
    * **`arm_angle_details`** (object): Contains the calculated angles and thresholds for the arm angle check.  
    * **`final_result`** (boolean): The same value as `is_target_pose_raw`, representing the combined result of all individual checks.

