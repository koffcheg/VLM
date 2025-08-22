#!/usr/bin/env python3
"""
Simple MQTT Sender Module for Video Intelligence
No arguments needed - just import and use!
Automatically loads configuration from config.yaml or environment variables.
"""

import os
import json
import yaml
import paho.mqtt.client as mqtt
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Global variables for state tracking
_mqtt_client = None
_store_id = None
_connected = False
_last_person_count = 0
_last_session_id = None
_initialized = False
_log_level = 'silent'  # Default to silent, will be set by run_inference.py

def _load_config():
    """Load configuration from config.yaml or environment variables"""
    global _store_id
    
    # Default values
    broker = '10.8.8.10'  #'localhost'
    port = 1883
    username = None
    password = None
    
    # Try to load from config.yaml - check multiple possible locations
    possible_paths = [
        '/home/checkout/eco_boutique_deploy/data/config.yaml',  # Deployment path
        '/workspace/config.yaml',  # Docker workspace path
        os.path.join(os.path.dirname(__file__), 'config.yaml')  # Local fallback
    ]
    
    config_path = None
    for path in possible_paths:
        if os.path.exists(path):
            config_path = path
            break
    if config_path and os.path.exists(config_path):
        try:
            _log(f"[SimpleMQTT] Loading config from: {config_path}")
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                mqtt_config = config.get('common', {}).get('mqtt', {})
                broker = '10.8.8.10'#mqtt_config.get('broker', broker)
                port = mqtt_config.get('port', port)
                _store_id = config.get('common', {}).get('store', {}).get('id', 'demo_40')
                # Handle env var references
                username = mqtt_config.get('username', '')
                password = mqtt_config.get('password', '')
                if username == '${MQTT_USERNAME}':
                    username = os.getenv('MQTT_USERNAME')
                if password == '${MQTT_PASSWORD}':
                    password = os.getenv('MQTT_PASSWORD')
            _log(f"[SimpleMQTT] Config loaded successfully - Store ID: {_store_id}, Broker: {broker}:{port}")
        except Exception as e:
            _log(f"[SimpleMQTT] Warning: Could not load config.yaml from {config_path}: {e}", 'error')
    else:
        _log("[SimpleMQTT] Warning: No config.yaml found, using environment variables and defaults", 'error')
    
    # Fallback to environment variables
    broker = '10.8.8.10' #os.getenv('MQTT_BROKER', broker)
    port = int(os.getenv('MQTT_PORT', port))
    username = os.getenv('MQTT_USERNAME', username)
    password = os.getenv('MQTT_PASSWORD', password)
    _store_id = os.getenv('STORE_ID', _store_id or 'jetson_store_001')
    
    return broker, port, username, password

def _log(message, level='info'):
    """Log message only if log level allows it"""
    if _log_level == 'debug' or (_log_level != 'silent' and level == 'error'):
        print(message)

def set_log_level(level):
    """Set the logging level for MQTT sender"""
    global _log_level
    _log_level = level
    _log(f"[SimpleMQTT] Log level set to: {level}")

def _on_connect(client, userdata, flags, rc):
    """MQTT connection callback"""
    global _connected
    if rc == 0:
        _connected = True
        _log(f"[SimpleMQTT] ✓ Connected to MQTT broker successfully (RC: {rc})")
        _log(f"[SimpleMQTT] Connection details - Client ID: {client._client_id}, Broker: {_load_config()[0]}:{_load_config()[1]}")
    else:
        _log(f"[SimpleMQTT] ✗ Connection failed with code {rc}", 'error')
        _log(f"[SimpleMQTT] RC codes: 0=success, 1=wrong protocol, 2=invalid client, 3=server unavailable, 4=bad credentials, 5=not authorized", 'error')

def _on_disconnect(client, userdata, rc):
    """MQTT disconnection callback"""
    global _connected
    _connected = False
    if rc != 0:
        _log(f"[SimpleMQTT] ⚠ Unexpected disconnect (code: {rc})", 'error')
    else:
        _log("[SimpleMQTT] Normal disconnect")

def _initialize_mqtt():
    """Initialize MQTT client if not already done"""
    global _mqtt_client, _initialized
    
    if _initialized:
        _log("[SimpleMQTT] Already initialized, skipping...")
        return
    
    try:
        broker, port, username, password = _load_config()
        _log(f"[SimpleMQTT] Initializing MQTT client...")
        _log(f"[SimpleMQTT] Config - Broker: {broker}, Port: {port}, Store ID: {_store_id}")
        _log(f"[SimpleMQTT] Auth - Username: {'SET' if username else 'NOT SET'}, Password: {'SET' if password else 'NOT SET'}")
        
        _mqtt_client = mqtt.Client(client_id=f"{_store_id}_video_intelligence_simple")
        
        if username and password:
            _mqtt_client.username_pw_set(username, password)
            _log("[SimpleMQTT] MQTT credentials configured")
        
        _mqtt_client.on_connect = _on_connect
        _mqtt_client.on_disconnect = _on_disconnect
        
        _log(f"[SimpleMQTT] Attempting connection to {broker}:{port}...")
        _mqtt_client.connect(broker, port, 60)
        _mqtt_client.loop_start()
        
        _initialized = True
        _log("[SimpleMQTT] MQTT client initialized successfully")
        
    except Exception as e:
        _log(f"[SimpleMQTT] ✗ Failed to initialize: {e}", 'error')
        if _log_level == 'debug':
            import traceback
            traceback.print_exc()
        _mqtt_client = None

def send_pipeline_output(pipeline_output: Dict[str, Any]):
    """
    Main function to send all detection results.
    Just pass the pipeline_output dictionary from run_inference.py
    
    Args:
        pipeline_output: The complete pipeline output dictionary
    """
    global _last_person_count, _last_session_id
    
    _log(f"[SimpleMQTT] 📤 send_pipeline_output() called")
    
    # Initialize on first use
    if not _initialized:
        _log("[SimpleMQTT] First call - initializing MQTT client...")
        _initialize_mqtt()
    
    # Check if we can send
    if not _mqtt_client:
        _log("[SimpleMQTT] ✗ No MQTT client available - cannot send messages", 'error')
        return
        
    if not _connected:
        _log("[SimpleMQTT] ⚠ MQTT client not connected - cannot send messages", 'error')
        return
        
    _log(f"[SimpleMQTT] ✓ MQTT client ready - connected: {_connected}")
    
    try:
        # Extract key information
        session_info = pipeline_output.get('session_info', {})
        session_id = session_info.get('session_id')
        session_active = session_info.get('is_active', False)
        
        primary_results = pipeline_output.get('pipeline_sources', {}).get('primary', {})
        people_detected = primary_results.get('detection_summary', {}).get('people_detected', 0)
        detected_people = primary_results.get('detected_people', [])
        
        _log(f"[SimpleMQTT] 📊 Processing pipeline data:")
        _log(f"[SimpleMQTT]   - Session ID: {session_id}")
        _log(f"[SimpleMQTT]   - Session Active: {session_active}")
        _log(f"[SimpleMQTT]   - People Detected: {people_detected}")
        _log(f"[SimpleMQTT]   - Previous People Count: {_last_person_count}")
        _log(f"[SimpleMQTT]   - Store ID: {_store_id}")
        
        # Send person detection events
        if people_detected > 0 and _last_person_count == 0:
            # Person entered frame
            topic = f'{_store_id}/vision/person_frame'
            message = {
                'person_in_frame': '1',
                'people_count': people_detected,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            _log(f"[SimpleMQTT] 👤 Person ENTERED frame - sending to {topic}")
            _send_message(topic, message)
        
        elif people_detected == 0 and _last_person_count > 0:
            # Person left frame
            topic = f'{_store_id}/vision/person_frame'
            message = {
                'person_in_frame': '0',
                'people_count': 0,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            _log(f"[SimpleMQTT] 👤 Person LEFT frame - sending to {topic}")
            _send_message(topic, message)
        
        # Check for gestures
        for person in detected_people:
            pose_analysis = person.get('pose_analysis', {})
            if pose_analysis.get('is_pose_confirmed', False):
                # Determine gesture type based on pose analysis details
                gesture_type = 'heart'  # Default to heart for backward compatibility
                
                # Check if this is a peace sign gesture (you can customize this logic)
                check_details = pose_analysis.get('check_details', {})
                if check_details.get('gesture_type'):
                    gesture_type = check_details['gesture_type']
                
                topic = f'{_store_id}/vision/events/gesture_detected'
                message = {
                    'event_type': 'gesture_detected',
                    'gesture': gesture_type,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'person_index': person.get('person_index', 0)
                }
                _log(f"[SimpleMQTT] ✋ {gesture_type.title()} gesture detected - sending to {topic}")
                _send_message(topic, message)
            
            # Check for near monitor or interaction zone
            active_zones = person.get('active_zones', [])
            if 'Customer Interaction Zone' in active_zones or 'Near Monitor' in active_zones:
                topic = f'{_store_id}/vision/near_monitor'
                message = {
                    'near_monitor': '1',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                _log(f"[SimpleMQTT] 🎯 Person in interaction zone - sending to {topic}")
                _send_message(topic, message)
        
        # Send full analysis data - THIS IS THE MAIN MESSAGE
        analysis_topic = f'{_store_id}/vision/analysis'
        _log(f"[SimpleMQTT] 📋 Sending FULL pipeline output to {analysis_topic}")
        _log(f"[SimpleMQTT] Pipeline data size: {len(str(pipeline_output))} characters")
        _send_message(analysis_topic, pipeline_output)
        
        # Update state
        _log(f"[SimpleMQTT] 🔄 Updating state: people count {_last_person_count} -> {people_detected}")
        _last_person_count = people_detected
        _last_session_id = session_id if session_active else None
        
        _log(f"[SimpleMQTT] ✅ Pipeline processing completed successfully")
        
    except Exception as e:
        _log(f"[SimpleMQTT] ❌ Error processing output: {e}", 'error')
        if _log_level == 'debug':
            import traceback
            traceback.print_exc()

def _send_message(topic: str, data: Dict[str, Any]):
    """Internal function to send MQTT message"""
    try:
        payload = json.dumps(data)
        payload_size = len(payload)
        
        _log(f"[SimpleMQTT] 📤 Publishing message:")
        _log(f"[SimpleMQTT]   Topic: {topic}")
        _log(f"[SimpleMQTT]   Payload size: {payload_size} bytes")
        _log(f"[SimpleMQTT]   QoS: 1")
        
        result = _mqtt_client.publish(topic, payload, qos=1)
        
        if result.rc == 0:
            _log(f"[SimpleMQTT] ✅ Message published successfully (Message ID: {result.mid})")
        else:
            _log(f"[SimpleMQTT] ❌ Failed to publish to {topic} (RC: {result.rc})", 'error')
            _log(f"[SimpleMQTT] RC meanings: 0=success, 1=incorrect protocol, 2=invalid client ID, 3=server unavailable, 4=bad username/password, 5=not authorized", 'error')
            
    except Exception as e:
        _log(f"[SimpleMQTT] ❌ Error sending message to {topic}: {e}", 'error')
        if _log_level == 'debug':
            import traceback
            traceback.print_exc()

def cleanup():
    """Clean up MQTT connection"""
    global _mqtt_client, _initialized
    
    if _mqtt_client:
        try:
            _mqtt_client.loop_stop()
            _mqtt_client.disconnect()
            print("[SimpleMQTT] Disconnected")
        except:
            pass
        _mqtt_client = None
        _initialized = False

# Optional: Auto-cleanup on exit
import atexit
atexit.register(cleanup)
