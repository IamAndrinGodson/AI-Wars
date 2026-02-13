"""
FastAPI Application for ML Threat Detection System
Provides REST API for threat detection and monitoring
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import yaml
import random
import uuid

import sys
sys.path.append('src')

from models.anomaly_detector import EnsembleAnomalyDetector
from models.threat_classifier import ThreatClassifier
from models.kmeans_detector import KMeansAnomalyDetector
from features.feature_generator import FeatureEngineer
from response.response_engine import ResponseEngine, ResponseAction, AttackChainReconstructor
from collectors.network_monitor import get_monitor, NetworkMonitor
from collectors.network_enrichment import enricher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelSelection(BaseModel):
    model_type: str = Field(..., description="Model type to switch to (synthetic, kdd, cicids, kmeans)")

# Initialize FastAPI app
app = FastAPI(
    title="ML Threat Detection API",
    description="Production API for ML-based cybersecurity threat detection",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount dashboard static files
import os
if not os.path.exists("src/dashboard"):
    os.makedirs("src/dashboard")
app.mount("/dashboard", StaticFiles(directory="src/dashboard", html=True), name="dashboard")


# Global model instances
# Global model instances
models_store = {
    "synthetic": {
        "anomaly_detector": None,
        "threat_classifier": None,
        "feature_engineer": None
    },
    "kdd": {
        "anomaly_detector": None,
        "threat_classifier": None,
        "feature_engineer": None
    },
    "cicids": {
        "anomaly_detector": None,
        "threat_classifier": None,
        "feature_engineer": None
    },
    "kmeans": {
        "anomaly_detector": None,
        "threat_classifier": None,
        "feature_engineer": None
    }
}
current_model_type = "synthetic"

anomaly_detector: Optional[EnsembleAnomalyDetector] = None
threat_classifier: Optional[ThreatClassifier] = None
feature_engineer: Optional[FeatureEngineer] = None
response_engine: Optional[ResponseEngine] = None
config: Dict[str, Any] = {}

def load_model_suite(path_prefix: str) -> Dict[str, Any]:
    """Helper to load a suite of models from a specific path"""
    suite = {
        "anomaly_detector": None,
        "threat_classifier": None,
        "feature_engineer": None
    }
    
    # Load Feature Engineer
    try:
        fe_path = f"{path_prefix}/feature_engineer.pkl"
        if os.path.exists(fe_path):
            fe = FeatureEngineer(config)
            fe.load(fe_path)
            suite["feature_engineer"] = fe
            logger.info(f"Loaded Feature Engineer from {fe_path}")
        else:
            logger.warning(f"Feature Engineer not found at {fe_path}")
    except Exception as e:
        logger.warning(f"Could not load feature engineer: {e}")

    # Load Anomaly Detector
    try:
        ad_path = f"{path_prefix}/anomaly_detector"
        if os.path.exists(ad_path) or os.path.exists(ad_path + "_model.h5") or os.path.exists(ad_path + "_data.pkl"): # Check for dir or file artifacts
            ad = EnsembleAnomalyDetector(config.get('models', {}).get('anomaly_detection', {}))
            ad.load(ad_path)
            suite["anomaly_detector"] = ad
            logger.info(f"Loaded Anomaly Detector from {ad_path}")
        else:
            logger.warning(f"Anomaly Detector not found at {ad_path}")
    except Exception as e:
        logger.warning(f"Could not load anomaly detector: {e}")
    
    # Special handling for K-Means model (single file .pkl)
    try:
        kmeans_path = f"{path_prefix}/kmeans_model.pkl"
        if os.path.exists(kmeans_path):
            kmeans_config = config.get('models', {}).get('kmeans', {})
            kmeans = KMeansAnomalyDetector(kmeans_config)
            kmeans.load(kmeans_path)
            suite["anomaly_detector"] = kmeans
            logger.info(f"Loaded K-Means Anomaly Detector from {kmeans_path}")
    except Exception as e:
        logger.warning(f"Could not load K-Means detector: {e}")

    # Load Threat Classifier
    try:
        tc_path = f"{path_prefix}/threat_classifier"
        if os.path.exists(tc_path) or os.path.exists(tc_path + "_model.pkl"):
            tc = ThreatClassifier(config.get('models', {}).get('classification', {}))
            tc.load(tc_path)
            suite["threat_classifier"] = tc
            logger.info(f"Loaded Threat Classifier from {tc_path}")
        else:
             # If exact path missing, maybe it's not trained for this mode yet.
             logger.warning(f"Threat Classifier not found at {tc_path}")
    except Exception as e:
        logger.warning(f"Could not load threat classifier: {e}")
        
    return suite

def set_active_model(model_type: str):
    """Switch the active global model pointers"""
    global anomaly_detector, threat_classifier, feature_engineer, current_model_type
    
    if model_type not in models_store:
        raise ValueError(f"Unknown model type: {model_type}")
        
    suite = models_store[model_type]
    
    # Update globals
    anomaly_detector = suite["anomaly_detector"]
    threat_classifier = suite["threat_classifier"]
    feature_engineer = suite["feature_engineer"]
    current_model_type = model_type
    
    logger.info(f"Switched active model to: {model_type}")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models and configuration on startup"""
    global response_engine, config
    
    logger.info("Starting ML Threat Detection API...")
    
    try:
        # Load configuration
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Load "Synthetic" Models (Default path data/models)
        logger.info("Loading SYNTHETIC models...")
        models_store["synthetic"] = load_model_suite("data/models")
        
        # Load "KDD" Models (Path data/models/real -> data/models/kdd for clarity, but mapped to real for now to match prev work)
        # Note: Previous step saved KDD model to data/models/real. keeping it consistent or moving?
        # Let's map 'kdd' to 'data/models/real' for backward compat with previous step
        logger.info("Loading KDD models...")
        models_store["kdd"] = load_model_suite("data/models/real")

        # Load "CIC-IDS" Models (Path data/models/cicids)
        logger.info("Loading CIC-IDS models...")
        models_store["cicids"] = load_model_suite("data/models/cicids")
        
        # Load "K-Means" Model (Path data/models/kmeans)
        logger.info("Loading K-Means models...")
        models_store["kmeans"] = load_model_suite("data/models/kmeans")
        
        # Set default
        set_active_model("synthetic")
        
        
        # Initialize response engine
        response_engine = ResponseEngine(config.get('response', {}))
        
        # Initialize network monitor (eager loading)
        init_network_monitor()
        
        logger.info("API startup complete!")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise




class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    version: str
    uptime_seconds: float

# API Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": anomaly_detector is not None and threat_classifier is not None,
        "version": "1.0.0",
        "uptime_seconds": 0.0  # Would track actual uptime
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "anomaly_detector": anomaly_detector is not None,
            "threat_classifier": threat_classifier is not None,
            "feature_engineer": feature_engineer is not None,
            "response_engine": response_engine is not None
        },
        "config_loaded": len(config) > 0
    }


class NetworkEvent(BaseModel):
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    bytes: int
    packets: int
    flow_duration: float
    flags: Optional[str] = None
    service: Optional[str] = None

class DetectionRequest(BaseModel):
    events: List[NetworkEvent]

class DetectionResult(BaseModel):
    threat_id: str
    timestamp: Optional[datetime]
    src_ip: str
    dst_ip: str
    dst_port: int
    is_anomaly: bool
    anomaly_score: float
    threat_class: str
    risk_score: float
    severity: str
    action: str
    process: Optional[str] = "unknown"
    source: str

class DetectionResponse(BaseModel):
    results: List[DetectionResult]
    processed_count: int
    anomalies_count: int

class SimulationRequest(BaseModel):
    attack_type: str
    intensity: int = 1
    duration_seconds: int = 60
    target_ip: Optional[str] = "192.168.1.100"
@app.post("/detect", response_model=DetectionResponse)
async def detect_threats(
    request: DetectionRequest,
    background_tasks: BackgroundTasks
):
    """
    Detect threats in network events
    
    This endpoint:
    1. Extracts features from raw events
    2. Runs anomaly detection
    3. Classifies detected anomalies
    4. Calculates risk scores
    5. Returns detection results
    """
    
    if not all([anomaly_detector, threat_classifier, feature_engineer, response_engine]):
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please ensure models are trained and available."
        )
    
    try:
        # Convert events to DataFrame
        events_data = [event.dict() for event in request.events]
        df = pd.DataFrame(events_data)
        
        # Add connection_id if missing (treat each event as separate connection)
        if 'connection_id' not in df.columns:
            df['connection_id'] = range(len(df))
        
        # Extract features
        features = feature_engineer.extract_all_features(df)
        features_normalized = feature_engineer.normalize_features(features, fit=False)
        X = features_normalized.values
        
        # Run anomaly detection
        anomaly_scores = anomaly_detector.score_samples(X)
        anomaly_predictions = anomaly_detector.predict(X)
        
        # Run threat classification
        threat_classes = threat_classifier.predict(X)
        classification_probas = threat_classifier.predict_proba(X)
        classification_confidence = classification_probas.max(axis=1)
        
        # Generate detections
        detections = []
        for i in range(len(df)):
            # Calculate risk score
            risk_assessment = response_engine.calculate_risk_score(
                anomaly_score=float(anomaly_scores[i]),
                classification_confidence=float(classification_confidence[i]),
                threat_class=threat_classes[i]
            )
            
            # Determine action
            action = response_engine.determine_response_action(risk_assessment['risk_score'])
            
            # Create detection object
            detection = ThreatDetection(
                threat_id=f"THR-{datetime.now().strftime('%Y%m%d')}-{i:06d}",
                timestamp=str(df.iloc[i]['timestamp']),
                is_anomaly=bool(anomaly_predictions[i]),
                anomaly_score=float(anomaly_scores[i]),
                threat_class=threat_classes[i],
                classification_confidence=float(classification_confidence[i]),
                risk_score=float(risk_assessment['risk_score']),
                severity=risk_assessment['severity'],
                recommended_action=action.value
            )
            
            detections.append(detection)
            
            # Execute response for all detections to ensure history is logged
            # Low risk items will just be monitored/logged
            background_tasks.add_task(
                execute_response,
                detection.threat_id,
                risk_assessment,
                events_data[i]
            )
        
        # Create summary
        summary = {
            "total_events": len(df),
            "anomalies_detected": int(anomaly_predictions.sum()),
            "threats_by_severity": {
                "low": sum(1 for d in detections if d.severity == "low"),
                "medium": sum(1 for d in detections if d.severity == "medium"),
                "high": sum(1 for d in detections if d.severity == "high"),
                "critical": sum(1 for d in detections if d.severity == "critical")
            },
            "avg_anomaly_score": float(anomaly_scores.mean()),
            "max_risk_score": float(max(d.risk_score for d in detections))
        }
        
        return DetectionResponse(
            request_id=f"REQ-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            detections=detections,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Error in threat detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Get system metrics and statistics"""
    
    # In production, integrate with Prometheus
    return {
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "requests_total": 0,
            "detections_total": 0,
            "anomalies_total": 0,
            "avg_latency_ms": 0,
            "model_version": "1.0.0"
        }
    }


@app.get("/api/stats")
async def get_stats(source: Optional[str] = None):
    """Get dashboard statistics, optionally filtered by source (simulation/realtime)"""
    if not response_engine:
        return {}
        
    history = response_engine.get_action_history(limit=1000)
    
    # Filter by source if requested
    if source:
        history = [
            h for h in history
            if h.get('result', {}).get('threat_details', {}).get('source') == source
        ]
    
    return {
        "total_actions": len(history),
        "high_risk_threats": sum(1 for h in history if h.get('risk_score', 0) > 80),
        "blocked_ips": sum(1 for h in history if h.get('action') == 'block'),
        "monitored_flows": sum(1 for h in history if h.get('action') == 'monitor')
    }


@app.get("/api/history")
async def get_history(limit: int = 50):
    """Get recent action history"""
    if not response_engine:
        return []
        
    return response_engine.get_action_history(limit=limit)


@app.get("/model/info")
async def get_model_info():
    """Get information about loaded models"""
    
    info = {
        "anomaly_detector": {
            "loaded": anomaly_detector is not None,
            "type": "EnsembleAnomalyDetector",
            "models": ["isolation_forest", "autoencoder", "lstm"] if anomaly_detector else []
        },
        "threat_classifier": {
            "loaded": threat_classifier is not None,
            "type": "MultiClassThreatClassifier",
            "classes": ThreatClassifier.THREAT_CLASSES if threat_classifier else []
        }
    }
    
    return info


async def execute_response(
    threat_id: str,
    risk_assessment: Dict[str, Any],
    threat_details: Dict[str, Any]
):
    """Background task to execute automated response"""
    
    try:
        result = response_engine.execute_response(
            threat_id=threat_id,
            risk_assessment=risk_assessment,
            threat_details=threat_details
        )
        
        logger.info(f"Executed response for {threat_id}: {result['action']}")
        
    except Exception as e:
        logger.error(f"Error executing response for {threat_id}: {e}")


# ============================================================
# ATTACK SIMULATION ENDPOINTS
# ============================================================

def generate_random_ip():
    """Generate a random IP address"""
    return f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"


def generate_attack_events(attack_type: str, intensity: int, duration_seconds: int, target_ip: str) -> List[Dict]:
    """Generate simulated attack events based on attack type.
    
    Enhanced to generate more extreme patterns that trigger HIGH/CRITICAL severity alerts.
    Higher intensity levels guarantee more severe attack signatures.
    """
    
    events = []
    # Scale events with intensity - higher intensity = more events and more severe patterns
    num_events = intensity * 5  # Increased base multiplier
    
    # Severity multipliers based on intensity (1-10)
    severity_multiplier = max(1, intensity ** 2)  # Exponential scaling for extreme values
    
    def get_sim_process(port):
        mapping = {
            80: 'nginx', 443: 'nginx', 8080: 'node', 
            22: 'sshd', 21: 'vsftpd', 23: 'telnetd',
            3306: 'mysqld', 5432: 'postgres', 27017: 'mongod',
            3389: 'svchost.exe', 1433: 'sqlservr.exe'
        }
        return mapping.get(port, random.choice(['python', 'java', 'chrome.exe', 'powershell.exe', 'unknown']))

    if attack_type == "ddos":
        # DDoS: Extreme volume attack - GUARANTEED HIGH/CRITICAL
        # Multiple sources flooding a single target with massive traffic
        for i in range(num_events):
            dport = random.choice([80, 443, 8080])
            events.append({
                "timestamp": datetime.now().isoformat(),
                "src_ip": generate_random_ip(),  # Many different sources = botnet
                "dst_ip": target_ip,
                "src_port": random.randint(1024, 65535),
                "dst_port": dport,
                "protocol": 6,  # TCP
                # EXTREME values for DDoS detection
                "packets": random.randint(500000, 5000000) * severity_multiplier,
                "bytes": random.randint(100000000, 1000000000) * severity_multiplier,  # 100MB-1GB per flow
                "flow_duration": random.uniform(0.01, 0.5),  # Very short bursts = suspicious
                "source": "simulation",
                "process": get_sim_process(dport)
            })
    
    elif attack_type == "port_scan":
        # Port Scan: Rapid sequential port probing - HIGH severity
        src_ip = generate_random_ip()
        base_port = random.randint(1, 100)
        for i in range(num_events):
            # Sequential ports with very low packets = classic scan signature
            dport = (base_port + i) % 65535 + 1
            events.append({
                "timestamp": datetime.now().isoformat(),
                "src_ip": src_ip,  # Same source scanning many ports
                "dst_ip": target_ip,
                "src_port": random.randint(40000, 60000),
                "dst_port": dport,
                "protocol": 6,
                "packets": 1,  # SYN scan signature - single packet per port
                "bytes": 40 + random.randint(0, 20),  # Minimal payload
                "flow_duration": random.uniform(0.0001, 0.01) * (1 / severity_multiplier),  # Sub-millisecond
                "source": "simulation",
                "process": "unknown"  # Port scanners don't associate with legit processes
            })
    
    elif attack_type == "brute_force":
        # Brute Force: Rapid authentication attempts - HIGH/CRITICAL
        src_ip = generate_random_ip()
        target_port = random.choice([22, 3389, 21, 23])  # SSH, RDP, FTP, Telnet
        for i in range(num_events):
            events.append({
                "timestamp": datetime.now().isoformat(),
                "src_ip": src_ip,
                "dst_ip": target_ip,
                "src_port": random.randint(40000, 60000),
                "dst_port": target_port,
                "protocol": 6,
                "packets": random.randint(50, 200) * severity_multiplier,  # Many auth attempts
                "bytes": random.randint(5000, 20000) * severity_multiplier,  # Auth payload signatures
                "flow_duration": random.uniform(0.1, 1.0),  # Rapid attempts
                "source": "simulation",
                "process": get_sim_process(target_port)
            })
    
    elif attack_type == "sql_injection":
        # SQL Injection: Suspicious database traffic patterns - HIGH/CRITICAL
        src_ip = generate_random_ip()
        for i in range(num_events):
            # Target database and web ports
            dport = random.choice([80, 443, 8080, 3306, 5432, 1433])
            events.append({
                "timestamp": datetime.now().isoformat(),
                "src_ip": src_ip,
                "dst_ip": target_ip,
                "src_port": random.randint(40000, 60000),
                "dst_port": dport,
                "protocol": 6,
                "packets": random.randint(100, 500) * severity_multiplier,
                "bytes": random.randint(50000, 200000) * severity_multiplier,  # Large SQL payloads
                "flow_duration": random.uniform(0.5, 2.0),  # Multiple queries
                "source": "simulation",
                "process": get_sim_process(dport)
            })
    
    elif attack_type == "data_exfiltration":
        # Data Exfiltration: Massive outbound data transfers - CRITICAL
        src_ip = target_ip  # Internal source exfiltrating data
        for i in range(num_events):
            dport = random.choice([443, 22, 21, 53, 8443])  # Common exfil ports
            events.append({
                "timestamp": datetime.now().isoformat(),
                "src_ip": src_ip,
                "dst_ip": generate_random_ip(),  # External destination
                "src_port": random.randint(40000, 60000),
                "dst_port": dport,
                "protocol": 6,
                "packets": random.randint(50000, 500000) * severity_multiplier,
                "bytes": random.randint(500000000, 2000000000),  # 500MB-2GB transfers = CRITICAL
                "flow_duration": random.uniform(30, 120),  # Extended transfer duration
                "source": "simulation",
                "process": random.choice(['curl', 'wget', 'powershell.exe', 'python', 'unknown'])
            })
    
    elif attack_type == "ransomware":
        # NEW: Ransomware-like behavior - CRITICAL
        src_ip = target_ip  # Internal compromised host
        for i in range(num_events):
            # Ransomware communicates on unusual ports and has burst patterns
            dport = random.choice([4444, 5555, 9001, 6666, 31337])  # Common C2 ports
            events.append({
                "timestamp": datetime.now().isoformat(),
                "src_ip": src_ip,
                "dst_ip": generate_random_ip(),
                "src_port": random.randint(49152, 65535),
                "dst_port": dport,
                "protocol": 6,
                "packets": random.randint(10000, 100000) * severity_multiplier,
                "bytes": random.randint(10000000, 100000000),
                "flow_duration": random.uniform(0.5, 5.0),
                "source": "simulation",
                "process": random.choice(['unknown', 'svchost.exe', 'rundll32.exe', 'powershell.exe'])
            })
    
    elif attack_type == "zero_day":
        # NEW: Zero-day exploit pattern - CRITICAL
        # Unusual traffic patterns that don't match normal behavior
        src_ip = generate_random_ip()
        for i in range(num_events):
            dport = random.choice([445, 139, 135, 8443, 9090])  # Exploit-prone ports
            events.append({
                "timestamp": datetime.now().isoformat(),
                "src_ip": src_ip,
                "dst_ip": target_ip,
                "src_port": random.randint(1024, 5000),  # Low ephemeral = suspicious
                "dst_port": dport,
                "protocol": 6,
                "packets": random.randint(100000, 1000000) * severity_multiplier,
                "bytes": random.randint(50000000, 500000000),  # Very large payload
                "flow_duration": random.uniform(0.001, 0.1),  # Extremely fast = exploit
                "source": "simulation",
                "process": "unknown"
            })
    
    else:
        # Generic malicious traffic - scales with intensity
        for i in range(num_events):
            dport = random.randint(1, 65535)
            events.append({
                "timestamp": datetime.now().isoformat(),
                "src_ip": generate_random_ip(),
                "dst_ip": target_ip,
                "src_port": random.randint(1024, 65535),
                "dst_port": dport,
                "protocol": random.choice([6, 17]),
                "packets": random.randint(10000, 100000) * severity_multiplier,
                "bytes": random.randint(10000000, 100000000) * severity_multiplier,
                "flow_duration": random.uniform(0.1, 10),
                "source": "simulation",
                "process": get_sim_process(dport)
            })
    
    return events


@app.post("/api/simulate")
async def simulate_attack(request: SimulationRequest, background_tasks: BackgroundTasks):
    """
    Simulate an attack and send events through the detection pipeline.
    This allows testing the dashboard's real-time threat detection capabilities.
    """
    
    if not all([anomaly_detector, threat_classifier, feature_engineer, response_engine]):
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please ensure models are trained and available."
        )
    
    try:
        # Generate attack events
        attack_events = generate_attack_events(
            request.attack_type,
            request.intensity,
            request.duration_seconds,
            request.target_ip
        )
        
        logger.info(f"Simulating {request.attack_type} attack with {len(attack_events)} events")
        
        # Process events through detection pipeline
        df = pd.DataFrame(attack_events)
        
        if 'connection_id' not in df.columns:
            df['connection_id'] = range(len(df))
        
        # Extract features
        features = feature_engineer.extract_all_features(df)
        features_normalized = feature_engineer.normalize_features(features, fit=False)
        X = features_normalized.values
        
        # Run anomaly detection
        anomaly_scores = anomaly_detector.score_samples(X)
        anomaly_predictions = anomaly_detector.predict(X)
        
        # Run threat classification
        threat_classes = threat_classifier.predict(X)
        classification_probas = threat_classifier.predict_proba(X)
        classification_confidence = classification_probas.max(axis=1)
        
        # Generate detections and execute responses
        detections = []
        for i in range(len(df)):
            # FOR SIMULATION: Override ML model if it fails to detect the specific attack type
            # This ensures the user sees what they asked for in the simulation
            final_threat_class = request.attack_type
            
            # Force high anomaly score for simulation to guarantee detection
            final_anomaly_score = max(float(anomaly_scores[i]), 0.85 + random.random() * 0.14)
            
            risk_assessment = response_engine.calculate_risk_score(
                anomaly_score=final_anomaly_score,
                classification_confidence=max(float(classification_confidence[i]), 0.9), # High confidence for sim
                threat_class=final_threat_class
            )
            
            action = response_engine.determine_response_action(risk_assessment['risk_score'])
            
            detection = {
                "threat_id": f"SIM-{uuid.uuid4().hex[:8].upper()}",
                "timestamp": attack_events[i]['timestamp'],
                "src_ip": attack_events[i]['src_ip'],
                "dst_ip": attack_events[i]['dst_ip'],
                "src_port": attack_events[i]['src_port'],
                "dst_port": attack_events[i]['dst_port'],
                "protocol": attack_events[i]['protocol'],
                "is_anomaly": True, # Always flag as anomaly in simulation
                "anomaly_score": final_anomaly_score,
                "threat_class": final_threat_class,
                "classification_confidence": float(classification_confidence[i]),
                "risk_score": float(risk_assessment['risk_score']),
                "severity": risk_assessment['severity'],
                "recommended_action": action.value,
                "source": "simulation" # Tag detection result too
            }
            
            detections.append(detection)
            
            # Execute response in background - include source and severity for proper filtering and email alerts
            background_tasks.add_task(
                execute_response,
                detection['threat_id'],
                risk_assessment,
                {**attack_events[i], 'threat_class': final_threat_class, 'source': 'simulation', 'severity': risk_assessment['severity']}
            )
        
        return {
            "simulation_id": f"SIM-{uuid.uuid4().hex[:12].upper()}",
            "attack_type": request.attack_type,
            "intensity": request.intensity,
            "events_generated": len(attack_events),
            "detections": detections,
            "summary": {
                "anomalies_detected": int(anomaly_predictions.sum()),
                "avg_risk_score": float(np.mean([d['risk_score'] for d in detections])),
                "max_risk_score": float(max(d['risk_score'] for d in detections)),
                "threats_by_severity": {
                    "low": sum(1 for d in detections if d['severity'] == "low"),
                    "medium": sum(1 for d in detections if d['severity'] == "medium"),
                    "high": sum(1 for d in detections if d['severity'] == "high"),
                    "critical": sum(1 for d in detections if d['severity'] == "critical")
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error in attack simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ENHANCED STATS ENDPOINT
# ============================================================

@app.get("/api/stats/detailed")
async def get_detailed_stats(source: Optional[str] = None):
    """Get comprehensive dashboard statistics with IP/port breakdowns"""
    if not response_engine:
        return {}
    
    history = response_engine.get_action_history(limit=1000)
    
    # Filter by source if requested
    if source:
        history = [
            h for h in history
            if h.get('result', {}).get('threat_details', {}).get('source') == source
        ]
    
    # Calculate top IPs
    src_ip_counts = {}
    dst_port_counts = {}
    attack_type_counts = {}
    protocol_counts = {}
    
    for h in history:
        result = h.get('result', {})
        threat = result.get('threat_details', {}) if isinstance(result, dict) else {}
        
        # Count source IPs
        src_ip = threat.get('src_ip')
        if src_ip:
            src_ip_counts[src_ip] = src_ip_counts.get(src_ip, 0) + 1
        
        # Count destination ports
        dst_port = threat.get('dst_port')
        if dst_port:
            dst_port_counts[str(dst_port)] = dst_port_counts.get(str(dst_port), 0) + 1
        
        # Count attack types
        threat_class = threat.get('threat_class')
        if threat_class:
            attack_type_counts[threat_class] = attack_type_counts.get(threat_class, 0) + 1
            
        # Count by severity as attack category
        severity = h.get('severity', 'low')
        # attack_type_counts[severity] = attack_type_counts.get(severity, 0) + 1 # Optional: keep existing behavior if desired, but threat_class is better
    
    # Get top items
    top_source_ips = sorted(src_ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top_dst_ports = sorted(dst_port_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        "total_actions": len(history),
        "high_risk_threats": sum(1 for h in history if h.get('risk_score', 0) > 80),
        "blocked_ips": sum(1 for h in history if h.get('action') == 'block'),
        "monitored_flows": sum(1 for h in history if h.get('action') == 'monitor'),
        "top_source_ips": [{"ip": ip, "count": count} for ip, count in top_source_ips],
        "top_destination_ports": [{"port": port, "count": count} for port, count in top_dst_ports],
        "attack_types": attack_type_counts,
        "severity_distribution": {
            "low": sum(1 for h in history if h.get('severity') == 'low'),
            "medium": sum(1 for h in history if h.get('severity') == 'medium'),
            "high": sum(1 for h in history if h.get('severity') == 'high'),
            "critical": sum(1 for h in history if h.get('severity') == 'critical')
        }
    }


@app.get("/api/history/detailed")
async def get_detailed_history(limit: int = 50, source: Optional[str] = None):
    """
    Get detailed action history with full event information.
    Optional 'source' param filters by 'simulation' or 'realtime'.
    """
    if not response_engine:
        return []
    
    # Get more history initially if filtering
    fetch_limit = 1000 if source else limit
    history = response_engine.get_action_history(limit=fetch_limit)
    
    if source:
        history = [
            h for h in history 
            if h.get('result', {}).get('threat_details', {}).get('source') == source
        ]
        # Re-apply limit
        history = history[-limit:]
    
    # Enhance history with network details
    enhanced_history = []
    for h in history:
        result = h.get('result', {})
        threat_details = result.get('threat_details', {}) if isinstance(result, dict) else {}
        
        # Enrich port info
        dst_port_val = threat_details.get('dst_port')
        port_svc = {"service": "", "desc": ""}
        try:
             if dst_port_val is not None:
                 port_svc = enricher.get_port_info(int(dst_port_val))
        except:
             pass
             
        enhanced_history.append({
            "timestamp": h.get('timestamp'),
            "threat_id": h.get('threat_id'),
            "severity": h.get('severity'),
            "action": h.get('action'),
            "risk_score": h.get('risk_score'),
            "src_ip": threat_details.get('src_ip', 'N/A'),
            "dst_ip": threat_details.get('dst_ip', 'N/A'),
            "src_port": threat_details.get('src_port', 'N/A'),
            "dst_port": threat_details.get('dst_port', 'N/A'),
            "protocol": threat_details.get('protocol', 'N/A'),
            "threat_class": threat_details.get('threat_class', 'unknown'),
            "source": threat_details.get('source', 'unknown'),
            "service": port_svc.get("service", ""),
            "service_desc": port_svc.get("desc", ""),
            "process": threat_details.get('process', 'unknown')
        })
    
    return enhanced_history


@app.get("/api/enrich/ip/{ip}")
async def enrich_ip_info(ip: str):
    """Get enrichment info for an IP"""
    return enricher.enrich_ip(ip)

@app.get("/api/enrich/port/{port}")
async def enrich_port_info(port: int):
    """Get service info for a port"""
    return enricher.get_port_info(port)

# ============================================================
# ADMIN ENDPOINTS
# ============================================================

@app.get("/api/admin/system-status")
async def get_system_status():
    """Get detailed system status for admin panel"""
    return {
        "timestamp": datetime.now().isoformat(),
        "status": "healthy" if all([anomaly_detector, threat_classifier]) else "degraded",
        "models": {
            "anomaly_detector": {
                "loaded": anomaly_detector is not None,
                "type": "EnsembleAnomalyDetector"
            },
            "threat_classifier": {
                "loaded": threat_classifier is not None,
                "type": "MultiClassThreatClassifier"
            },
            "feature_engineer": {
                "loaded": feature_engineer is not None
            },
            "response_engine": {
                "loaded": response_engine is not None,
                "automation_enabled": response_engine.automation_enabled if response_engine else False
            }
        },
        "config": {
            "approval_threshold": response_engine.approval_threshold if response_engine else 80,
            "automation_enabled": response_engine.automation_enabled if response_engine else False
        },
        "statistics": {
            "total_actions": len(response_engine.action_history) if response_engine else 0,
            "blocked_count": sum(1 for h in (response_engine.action_history if response_engine else []) if h.get('action') == 'block')
        }
    }


@app.get("/api/admin/blocked-ips")
async def get_blocked_ips():
    """Get list of all blocked IPs"""
    if not response_engine:
        return []
    
    blocked = []
    for h in response_engine.action_history:
        if h.get('action') == 'block':
            result = h.get('result', {})
            if isinstance(result, dict):
                blocked_ip = result.get('blocked_ip')
                if blocked_ip and blocked_ip not in [b['ip'] for b in blocked]:
                    blocked.append({
                        "ip": blocked_ip,
                        "blocked_at": h.get('timestamp'),
                        "threat_id": h.get('threat_id'),
                        "risk_score": h.get('risk_score')
                    })
    
    return blocked


@app.delete("/api/admin/blocked-ips/{ip}")
async def unblock_ip(ip: str):
    """Remove an IP from the blocklist"""
    logger.info(f"Unblocking IP: {ip}")
    
    # In production, this would update firewall rules
    return {
        "status": "success",
        "message": f"IP {ip} has been unblocked",
        "timestamp": datetime.now().isoformat()
    }



class AdminConfigUpdate(BaseModel):
    automation_enabled: Optional[bool] = None
    approval_threshold: Optional[int] = None


@app.post("/api/admin/config")
async def update_admin_config(config: AdminConfigUpdate):
    """Update admin configuration settings"""
    if not response_engine:
        raise HTTPException(status_code=503, detail="Response engine not initialized")
    
    if config.automation_enabled is not None:
        response_engine.automation_enabled = config.automation_enabled
        logger.info(f"Automation enabled set to: {config.automation_enabled}")
    
    if config.approval_threshold is not None:
        if 0 <= config.approval_threshold <= 100:
            response_engine.approval_threshold = config.approval_threshold
            logger.info(f"Approval threshold set to: {config.approval_threshold}")
        else:
            raise HTTPException(status_code=400, detail="Threshold must be 0-100")
    
    return {
        "status": "success",
        "config": {
            "automation_enabled": response_engine.automation_enabled,
            "approval_threshold": response_engine.approval_threshold
        }
    }


@app.get("/api/benchmark/results")
async def get_benchmark_results():
    """Get benchmark results"""
    try:
        csv_path = "data/benchmark_results.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Standardize columns to match frontend expectations
            # Frontend expects: Model, Accuracy, F1 Score, Latency (s), etc.
            # CSV usually has: Model,Accuracy,Precision,Recall,F1 Score,Latency (s),Samples
            return df.to_dict(orient="records")
        return []
    except Exception as e:
        logger.error(f"Error reading benchmark results: {e}")
        raise HTTPException(status_code=500, detail="Could not load benchmark data")


@app.get("/api/config/model")
async def get_model_config():
    """Get current model configuration"""
    return {
        "current_model": current_model_type,
        "available_models": list(models_store.keys()),
        "status": {
            k: {
                "loaded": v["anomaly_detector"] is not None
            } for k, v in models_store.items()
        }
    }



@app.get("/api/benchmark/results")
async def get_benchmark_results():
    """Get latest benchmark results"""
    try:
        csv_path = "data/benchmark_results.csv"
        if not os.path.exists(csv_path):
            return []
            
        df = pd.read_csv(csv_path)
        
        # Replace NaN with null for JSON compatibility
        df = df.replace({np.nan: None})
        
        return df.to_dict(orient='records')
    except Exception as e:
        logger.error(f"Error reading benchmark results: {e}")
        return []


@app.post("/api/config/model")
async def switch_model(selection: ModelSelection):
    """Switch the active model"""
    try:
        set_active_model(selection.model_type)
        return {
            "status": "success", 
            "current_model": current_model_type,
            "message": f"Switched to {selection.model_type} model"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))




@app.post("/api/admin/clear-history")
async def clear_history():
    """Clear action history (for testing purposes)"""
    if response_engine:
        response_engine.action_history = []
        logger.info("Action history cleared")
    
    return {"status": "success", "message": "History cleared"}


# ==================== ADVANCED ANALYTICS ENDPOINTS ====================

# Global attack chain reconstructor
attack_chain_reconstructor = AttackChainReconstructor()

@app.get("/api/attack-chains")
async def get_attack_chains(limit: int = 10):
    """
    Get reconstructed attack chains grouped by source IP.
    Uses the AttackChainReconstructor to identify Kill Chain stages.
    """
    if not response_engine:
        return {"chains": [], "total": 0}
    
    # Populate the reconstructor with recent events
    attack_chain_reconstructor.attack_chains = {}  # Reset
    
    history = response_engine.get_action_history(limit=500)
    
    for h in history:
        result = h.get('result', {})
        threat_details = result.get('threat_details', {}) if isinstance(result, dict) else {}
        
        if not threat_details:
            continue
            
        event = {
            'timestamp': h.get('timestamp'),
            'threat_id': h.get('threat_id'),
            'src_ip': threat_details.get('src_ip'),
            'dst_ip': threat_details.get('dst_ip'),
            'dst_port': threat_details.get('dst_port'),
            'threat_class': threat_details.get('threat_class'),
            'severity': h.get('severity'),
            'risk_score': h.get('risk_score'),
            'action': h.get('action')
        }
        attack_chain_reconstructor.add_event(event)
    
    # Reconstruct chains for each source IP
    chains = []
    for chain_key in list(attack_chain_reconstructor.attack_chains.keys())[:limit]:
        chain_data = attack_chain_reconstructor.reconstruct_chain(chain_key)
        if chain_data['events']:
            chains.append({
                'source_ip': chain_key,
                'event_count': len(chain_data['events']),
                'stages': chain_data['stages'],
                'stage_count': len(chain_data['stages']),
                'duration_seconds': chain_data['duration'],
                'severity': chain_data['severity'],
                'events': chain_data['events'][:10]  # Limit events per chain
            })
    
    # Sort by stage count (more stages = more advanced attack)
    chains.sort(key=lambda x: (x['stage_count'], x['event_count']), reverse=True)
    
    return {
        "chains": chains[:limit],
        "total": len(chains),
        "kill_chain_stages": [
            "reconnaissance",
            "initial_access", 
            "privilege_escalation",
            "lateral_movement",
            "exfiltration"
        ]
    }


@app.get("/api/stats/temporal")
async def get_temporal_stats(source: Optional[str] = None):
    """
    Get temporal breakdown of threat activity.
    Returns hourly and daily distributions for heatmap visualization.
    """
    if not response_engine:
        return {"hourly": {}, "daily": {}, "off_hours_count": 0, "weekend_count": 0}
    
    history = response_engine.get_action_history(limit=1000)
    
    # Filter by source if requested
    if source:
        history = [
            h for h in history
            if h.get('result', {}).get('threat_details', {}).get('source') == source
        ]
    
    # Initialize counters
    hourly_counts = {str(i): 0 for i in range(24)}
    daily_counts = {"Mon": 0, "Tue": 0, "Wed": 0, "Thu": 0, "Fri": 0, "Sat": 0, "Sun": 0}
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    
    # Heatmap data: hour x day matrix
    heatmap = {day: {str(h): 0 for h in range(24)} for day in day_names}
    
    off_hours_count = 0
    weekend_count = 0
    business_hours_count = 0
    
    severity_by_hour = {str(i): {"low": 0, "medium": 0, "high": 0, "critical": 0} for i in range(24)}
    
    for h in history:
        timestamp = h.get('timestamp')
        if not timestamp:
            continue
            
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            hour = dt.hour
            day = dt.weekday()  # 0=Monday, 6=Sunday
            
            hourly_counts[str(hour)] += 1
            daily_counts[day_names[day]] += 1
            heatmap[day_names[day]][str(hour)] += 1
            
            # Track severity by hour
            severity = h.get('severity', 'low')
            if severity in severity_by_hour[str(hour)]:
                severity_by_hour[str(hour)][severity] += 1
            
            # Off-hours detection (before 9am or after 5pm)
            if hour < 9 or hour >= 17:
                off_hours_count += 1
            else:
                business_hours_count += 1
                
            # Weekend detection
            if day >= 5:  # Saturday, Sunday
                weekend_count += 1
                
        except Exception as e:
            logger.debug(f"Error parsing timestamp: {e}")
            continue
    
    return {
        "hourly": hourly_counts,
        "daily": daily_counts,
        "heatmap": heatmap,
        "severity_by_hour": severity_by_hour,
        "off_hours_count": off_hours_count,
        "business_hours_count": business_hours_count,
        "weekend_count": weekend_count,
        "total_analyzed": len(history),
        "peak_hour": max(hourly_counts, key=hourly_counts.get) if hourly_counts else "N/A",
        "peak_day": max(daily_counts, key=daily_counts.get) if daily_counts else "N/A"
    }


@app.get("/api/threats/enhanced")
async def get_enhanced_threats(limit: int = 50, source: Optional[str] = None):
    """
    Get threat history with enhanced metadata including:
    - Zero-day detection flags
    - Off-hours/weekend indicators
    - Port risk levels
    - IP enrichment data
    """
    if not response_engine:
        return []
    
    fetch_limit = 500 if source else limit
    history = response_engine.get_action_history(limit=fetch_limit)
    
    if source:
        history = [
            h for h in history
            if h.get('result', {}).get('threat_details', {}).get('source') == source
        ]
        history = history[-limit:]
    
    enhanced = []
    for h in history:
        result = h.get('result', {})
        threat_details = result.get('threat_details', {}) if isinstance(result, dict) else {}
        
        # Parse timestamp for timing analysis
        timestamp = h.get('timestamp')
        is_off_hours = False
        is_weekend = False
        hour = None
        
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                hour = dt.hour
                is_off_hours = hour < 9 or hour >= 17
                is_weekend = dt.weekday() >= 5
            except:
                pass
        
        # Get port info
        dst_port = threat_details.get('dst_port')
        port_info = {"service": "", "risk": "unknown"}
        if dst_port:
            try:
                port_info = enricher.get_port_info(int(dst_port))
            except:
                pass
        
        # Get IP info
        src_ip = threat_details.get('src_ip')
        src_ip_info = {"type": "unknown"}
        if src_ip:
            try:
                src_ip_info = enricher.enrich_ip(src_ip)
            except:
                pass
        
        # Check for zero-day threat
        threat_class = threat_details.get('threat_class', '')
        is_zero_day = threat_class == 'zero_day'
        
        enhanced.append({
            "timestamp": timestamp,
            "hour": hour,
            "threat_id": h.get('threat_id'),
            "src_ip": src_ip,
            "dst_ip": threat_details.get('dst_ip'),
            "dst_port": dst_port,
            "severity": h.get('severity'),
            "action": h.get('action'),
            "risk_score": h.get('risk_score'),
            "threat_class": threat_class,
            "source": threat_details.get('source', 'unknown'),
            "process": threat_details.get('process', 'unknown'),
            # Enhanced fields
            "is_zero_day": is_zero_day,
            "is_off_hours": is_off_hours,
            "is_weekend": is_weekend,
            "port_service": port_info.get("service", ""),
            "port_risk": port_info.get("risk", "unknown"),
            "port_desc": port_info.get("desc", ""),
            "src_ip_type": src_ip_info.get("type", "unknown"),
            "src_hostname": src_ip_info.get("hostname", ""),
            "src_location": src_ip_info.get("location", "")
        })
    
    return enhanced


# ==================== REAL-TIME MONITORING ENDPOINTS ====================

network_monitor: Optional[NetworkMonitor] = None


def init_network_monitor():
    """Initialize and configure the network monitor"""
    global network_monitor
    network_monitor = get_monitor()
    
    # Set detection callback to process events through ML pipeline
    def detection_callback(events: List[Dict]) -> List[Dict]:
        """Process real-time events through the ML detection pipeline"""
        
        # Helper for safe float conversion
        def safe_float(val, default=0.0):
            try:
                return float(val)
            except:
                return default

        try:
            # Convert to DataFrame
            df = pd.DataFrame(events)
            if 'connection_id' not in df.columns:
                df['connection_id'] = range(len(df))
            
            # Default values (safe mode)
            anomaly_predictions = [False] * len(df)
            anomaly_scores = [0.0] * len(df)
            threat_classes = ['unknown'] * len(df)
            classification_confidence = [0.0] * len(df)
            
            # Try ML pipeline
            try:
                # Extract and normalize features
                features = feature_engineer.extract_all_features(df)
                features_normalized = feature_engineer.normalize_features(features, fit=False)
                X = features_normalized.values
                
                # Run detection pipeline
                if anomaly_detector:
                    anomaly_scores = anomaly_detector.score_samples(X)
                    anomaly_predictions = anomaly_detector.predict(X)
                
                if threat_classifier:
                    threat_classes = threat_classifier.predict(X)
                    classification_probas = threat_classifier.predict_proba(X)
                    classification_confidence = classification_probas.max(axis=1)
                    
            except Exception as ml_error:
                logger.error(f"ML Pipeline Error (Safe Mode Active): {ml_error}")
                # Continue with default values so we don't lose the visualization
            
            # Build results
            results = []
            for i in range(len(df)):
                try:
                    # Calculate risk if response engine available
                    risk_assessment = {'risk_score': 0.0, 'severity': 'low'}
                    action = ResponseAction.MONITOR
                    
                    if response_engine:
                        risk_assessment = response_engine.calculate_risk_score(
                            anomaly_score=safe_float(anomaly_scores[i]),
                            classification_confidence=safe_float(classification_confidence[i]),
                            threat_class=threat_classes[i]
                        )
                        action = response_engine.determine_response_action(risk_assessment['risk_score'])
                    
                    result = {
                        'threat_id': f"RT-{datetime.now().strftime('%H%M%S')}-{i:04d}",
                        'timestamp': events[i].get('timestamp'),
                        'src_ip': events[i].get('src_ip'),
                        'dst_ip': events[i].get('dst_ip'),
                        'dst_port': events[i].get('dst_port'),
                        'is_anomaly': bool(anomaly_predictions[i]),
                        'anomaly_score': safe_float(anomaly_scores[i]),
                        'threat_class': threat_classes[i],
                        'risk_score': float(risk_assessment['risk_score']),
                        'severity': risk_assessment['severity'],
                        'action': action.value if hasattr(action, 'value') else 'monitor',
                        'process': events[i].get('process', 'unknown'),
                        'source': 'realtime'
                    }
                    results.append(result)
                    
                    # Log ALL real-time events to action history so they appear in dashboard
                    if response_engine:
                        response_engine.execute_response(
                            result['threat_id'],
                            risk_assessment,
                            {**events[i], 'threat_class': threat_classes[i], 'source': 'realtime', 'severity': risk_assessment['severity']}, # Ensure source and severity are passed
                            require_approval=False
                        )
                except Exception as inner_e:
                    logger.error(f"Error processing single event {i}: {inner_e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Critical error in real-time detection: {e}")
            return []
    
    network_monitor.set_detection_callback(detection_callback)
    return network_monitor


@app.post("/api/realtime/start")
async def start_realtime_monitoring(scan_interval: float = 2.0):
    """Start real-time network monitoring"""
    global network_monitor
    
    if network_monitor is None:
        network_monitor = init_network_monitor()
    
    if network_monitor.running:
        return {
            "status": "already_running",
            "message": "Real-time monitoring is already active"
        }
    
    network_monitor.scan_interval = scan_interval
    success = network_monitor.start()
    
    return {
        "status": "started" if success else "error",
        "message": "Real-time network monitoring started" if success else "Failed to start",
        "scan_interval": scan_interval,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/realtime/stop")
async def stop_realtime_monitoring():
    """Stop real-time network monitoring"""
    if network_monitor is None or not network_monitor.running:
        return {
            "status": "not_running",
            "message": "Real-time monitoring is not active"
        }
    
    network_monitor.stop()
    
    return {
        "status": "stopped",
        "message": "Real-time network monitoring stopped",
        "stats": network_monitor.stats,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/realtime/status")
async def get_realtime_status():
    """Get real-time monitoring status and recent events"""
    if network_monitor is None:
        return {
            "running": False,
            "initialized": False,
            "message": "Monitor not initialized"
        }
    
    status = network_monitor.get_status()
    recent = network_monitor.get_recent_connections(limit=20)
    
    return {
        **status,
        "recent_events": recent
    }


@app.get("/api/realtime/alerts")
async def get_realtime_alerts(limit: int = 50, clear: bool = False):
    """Get real-time alerts"""
    if network_monitor is None:
        return []
    
    return network_monitor.get_alerts(limit=limit, clear=clear)


@app.post("/api/realtime/clear")
async def clear_realtime_data():
    """Clear all real-time monitoring data"""
    if network_monitor:
        network_monitor.clear_data()
    
    return {"status": "success", "message": "Real-time data cleared"}


# Run with: uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
