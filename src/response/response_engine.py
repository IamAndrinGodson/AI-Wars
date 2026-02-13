"""
Automated Response Engine
Implements tiered response actions based on threat severity
"""

import logging
from typing import Dict, Any, List, Optional
from enum import Enum
from datetime import datetime
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time

logger = logging.getLogger(__name__)


class ResponseAction(Enum):
    """Available response actions"""
    MONITOR = "monitor"
    ALERT = "alert"
    ISOLATE = "isolate"
    BLOCK = "block"
    QUARANTINE = "quarantine"
    TERMINATE = "terminate"


class ThreatSeverity(Enum):
    """Threat severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ResponseEngine:
    """Main automated response orchestrator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.automation_enabled = config.get('automation', {}).get('enabled', True)
        self.approval_threshold = config.get('automation', {}).get('require_approval_threshold', 80)
        self.actions_config = config.get('actions', {})
        
        # Debug logging
        logger.info(f"DEBUG ResponseEngine __init__: config keys: {config.keys()}")
        logger.info(f"DEBUG ResponseEngine __init__: actions_config: {self.actions_config}")
        logger.info(f"DEBUG ResponseEngine __init__: notifications config: {config.get('notifications', {})}")
        
        self.action_history: List[Dict[str, Any]] = []
        
        # Rate limiting for emails
        self.last_email_sent_time = 0
        self.email_rate_limit_seconds = 60  # Minimum seconds between emails
        
    def calculate_risk_score(
        self,
        anomaly_score: float,
        classification_confidence: float,
        threat_class: str,
        asset_criticality: float = 5.0,
        user_risk_score: float = 1.0
    ) -> Dict[str, Any]:
        """Calculate comprehensive risk score"""
        
        # Base threat severity by class
        threat_severity_map = {
            'normal': 0,
            'dos': 60,
            'ddos': 70,
            'port_scan': 40,
            'brute_force': 65,
            'sql_injection': 85,
            'xss': 75,
            'malware': 90,
            'data_exfiltration': 95,
            'lateral_movement': 80,
            'privilege_escalation': 85,
            'ransomware': 100,
            'phishing': 70,
            'botnet': 80,
            'zero_day': 95
        }
        
        base_severity = threat_severity_map.get(threat_class, 50)
        
        # Multi-factor risk calculation
        risk_score = (
            base_severity * 0.5 +  # Increased weight for Threat Type (was 0.4)
            anomaly_score * 100 * 0.2 +  # Anomaly detection (20%)
            classification_confidence * 100 * 0.2 +  # Classification confidence (20%)
            asset_criticality * 10 * 0.05 +  # Asset value (5%)
            user_risk_score * 10 * 0.05  # User risk profile (5%)
        )
        
        # Clamp to 0-100
        risk_score = max(0, min(100, risk_score))
        
        # Determine severity
        if risk_score < 40:
            severity = ThreatSeverity.LOW
        elif risk_score < 60:
            severity = ThreatSeverity.MEDIUM
        elif risk_score < 80:
            severity = ThreatSeverity.HIGH
        else:
            severity = ThreatSeverity.CRITICAL
        
        return {
            'risk_score': risk_score,
            'severity': severity.value,
            'factors': {
                'base_severity': base_severity,
                'anomaly_score': anomaly_score,
                'classification_confidence': classification_confidence,
                'asset_criticality': asset_criticality,
                'user_risk_score': user_risk_score
            }
        }
    
    def determine_response_action(self, risk_score: float) -> ResponseAction:
        """Determine appropriate response action based on risk score"""
        
        for action_name, action_config in self.actions_config.items():
            risk_range = action_config.get('risk_score_range', [0, 100])
            
            if risk_range[0] <= risk_score <= risk_range[1]:
                return ResponseAction(action_config.get('action', 'monitor'))
        
        # Default to monitoring
        return ResponseAction.MONITOR
    
    def execute_response(
        self,
        threat_id: str,
        risk_assessment: Dict[str, Any],
        threat_details: Dict[str, Any],
        require_approval: bool = False
    ) -> Dict[str, Any]:
        """Execute automated response action"""
        
        risk_score = risk_assessment['risk_score']
        severity = risk_assessment.get('severity', 'low')
        action = self.determine_response_action(risk_score)
        
        logger.info(f"DEBUG execute_response: threat_id={threat_id}, risk={risk_score:.1f}, severity={severity}, action={action.value}")
        
        # ALWAYS send email notifications for high/critical severity threats
        # This happens BEFORE any approval/queuing logic
        if severity in ['high', 'critical']:
            logger.info(f"DEBUG: High/Critical severity detected ({severity}), sending email notification")
            alert_data = self._create_alert_data(threat_details)
            self._send_notifications(alert_data)
        
        # Check if approval required
        needs_approval = (
            require_approval or 
            risk_score >= self.approval_threshold
        )
        
        if needs_approval and self.automation_enabled:
            logger.warning(
                f"Threat {threat_id} requires approval (risk: {risk_score:.1f}). "
                f"Queuing for review."
            )
            return self._queue_for_approval(threat_id, risk_assessment, threat_details, action)
        
        # Execute action
        if self.automation_enabled:
            result = self._execute_action(action, threat_details)
        else:
            logger.info(f"Automation disabled. Would execute: {action.value}")
            result = {'status': 'dry_run', 'action': action.value}
        
        # Add threat_details to result for dashboard access
        result['threat_details'] = threat_details
        
        # Log action
        self._log_action(threat_id, action, risk_assessment, result)
        
        return {
            'threat_id': threat_id,
            'action': action.value,
            'risk_score': risk_score,
            'severity': risk_assessment['severity'],
            'executed': self.automation_enabled,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
    
    def _execute_action(
        self,
        action: ResponseAction,
        threat_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute specific response action"""
        
        logger.info(f"Executing action: {action.value}")
        logger.info(f"DEBUG: threat_details keys: {threat_details.keys() if threat_details else 'None'}")
        logger.info(f"DEBUG: threat_details severity: {threat_details.get('severity') if threat_details else 'None'}")
        
        if action == ResponseAction.MONITOR:
            return self._action_monitor(threat_details)
        elif action == ResponseAction.ALERT:
            return self._action_alert(threat_details)
        elif action == ResponseAction.ISOLATE:
            return self._action_isolate(threat_details)
        elif action == ResponseAction.BLOCK:
            return self._action_block(threat_details)
        elif action == ResponseAction.QUARANTINE:
            return self._action_quarantine(threat_details)
        elif action == ResponseAction.TERMINATE:
            return self._action_terminate(threat_details)
        else:
            logger.error(f"Unknown action: {action}")
            return {'status': 'error', 'message': 'Unknown action'}
    
    def _action_monitor(self, threat_details: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor and log the threat"""
        logger.info(f"Monitoring threat: {threat_details.get('threat_class')}")
        
        # Enhanced monitoring for suspicious activity
        return {
            'status': 'success',
            'action': 'monitor',
            'monitoring_enabled': True
        }
    
    def _action_alert(self, threat_details: Dict[str, Any]) -> Dict[str, Any]:
        """Send alert to security team"""
        logger.warning(f"ALERT: {threat_details.get('threat_class')} detected")
        
        # In production, integrate with:
        # - Email/SMS notifications
        # - Slack/Teams
        # - PagerDuty
        # - SIEM alerts
        
        alert_data = self._create_alert_data(threat_details)
        
        # Send to notification channels
        self._send_notifications(alert_data)
        
        return {
            'status': 'success',
            'action': 'alert',
            'alert_sent': True,
            'channels': ['email', 'slack', 'siem']
        }
    
    def _create_alert_data(self, threat_details: Dict[str, Any]) -> Dict[str, Any]:
        """Create standardized alert data structure"""
        return {
            'timestamp': datetime.now().isoformat(),
            'threat_class': threat_details.get('threat_class'),
            'source_ip': threat_details.get('src_ip'),
            'destination': threat_details.get('dst_ip'),
            'severity': threat_details.get('severity'),
            'description': self._generate_alert_description(threat_details)
        }
    
    def _action_isolate(self, threat_details: Dict[str, Any]) -> Dict[str, Any]:
        """Isolate affected system from network"""
        logger.critical(f"ISOLATING: {threat_details.get('src_ip')}")
        
        # In production, integrate with:
        # - SDN controllers
        # - Firewall APIs
        # - NAC systems
        # - Switch port isolation
        
        source_ip = threat_details.get('src_ip')
        
        # Simulate network isolation
        isolation_result = self._isolate_network_segment(source_ip)
        
        # Collect forensic evidence
        self._collect_forensic_evidence(threat_details)
        
        # Send notifications
        alert_data = self._create_alert_data(threat_details)
        self._send_notifications(alert_data)

        return {
            'status': 'success',
            'action': 'isolate',
            'isolated_ip': source_ip,
            'isolation_result': isolation_result,
            'forensics_collected': True
        }
    
    def _action_block(self, threat_details: Dict[str, Any]) -> Dict[str, Any]:
        """Block malicious traffic"""
        logger.critical(f"BLOCKING: {threat_details.get('src_ip')}")
        
        # In production, integrate with:
        # - Firewall rules
        # - IPS/IDS
        # - Web Application Firewall
        # - DDoS mitigation
        
        source_ip = threat_details.get('src_ip')
        
        # Add to blocklist
        block_result = self._add_to_blocklist(source_ip)
        
        # Send notifications
        alert_data = self._create_alert_data(threat_details)
        self._send_notifications(alert_data)

        return {
            'status': 'success',
            'action': 'block',
            'blocked_ip': source_ip,
            'block_result': block_result,
            'firewall_updated': True
        }
    
    def _action_quarantine(self, threat_details: Dict[str, Any]) -> Dict[str, Any]:
        """Quarantine affected files/systems"""
        logger.critical(f"QUARANTINING: {threat_details}")
        
        # In production:
        # - Move files to quarantine
        # - Disable user accounts
        # - Snapshot systems for analysis
        
        # Send notifications
        alert_data = self._create_alert_data(threat_details)
        self._send_notifications(alert_data)

        return {
            'status': 'success',
            'action': 'quarantine',
            'quarantine_location': '/quarantine',
            'snapshot_created': True
        }
    
    def _action_terminate(self, threat_details: Dict[str, Any]) -> Dict[str, Any]:
        """Terminate malicious processes/connections"""
        logger.critical(f"TERMINATING: {threat_details}")
        
        # In production:
        # - Kill processes
        # - Terminate sessions
        # - Shutdown systems if necessary
        
        # Send notifications
        alert_data = self._create_alert_data(threat_details)
        self._send_notifications(alert_data)

        return {
            'status': 'success',
            'action': 'terminate',
            'terminated': True
        }
    
    def _queue_for_approval(
        self,
        threat_id: str,
        risk_assessment: Dict[str, Any],
        threat_details: Dict[str, Any],
        recommended_action: ResponseAction
    ) -> Dict[str, Any]:
        """Queue high-risk actions for human approval"""
        
        approval_request = {
            'threat_id': threat_id,
            'timestamp': datetime.now().isoformat(),
            'risk_assessment': risk_assessment,
            'threat_details': threat_details,
            'recommended_action': recommended_action.value,
            'status': 'pending_approval'
        }
        
        # In production: Send to approval workflow system
        logger.info(f"Approval request created for threat {threat_id}")
        
        return approval_request
    
    def _generate_alert_description(self, threat_details: Dict[str, Any]) -> str:
        """Generate human-readable alert description"""
        threat_class = threat_details.get('threat_class', 'unknown')
        src_ip = threat_details.get('src_ip', 'unknown')
        dst_ip = threat_details.get('dst_ip', 'unknown')
        
        return (
            f"Detected {threat_class} attack from {src_ip} targeting {dst_ip}. "
            f"Immediate investigation recommended."
        )
    
    def _send_notifications(self, alert_data: Dict[str, Any]):
        """Send notifications through various channels"""
        # Simulated - in production, integrate with actual notification services
        logger.info(f"Sending notifications: {json.dumps(alert_data, indent=2)}")
        
        # Check if email notifications are enabled
        email_config = self.config.get('notifications', {}).get('email', {})
        logger.info(f"DEBUG: Email config found: {bool(email_config)}, enabled: {email_config.get('enabled', False)}")
        
        if not email_config.get('enabled', False):
            logger.info("DEBUG: Email notifications disabled in config, skipping email")
            return

        # Check rate limit
        current_time = time.time()
        if current_time - self.last_email_sent_time < self.email_rate_limit_seconds:
            logger.info(f"DEBUG: Email rate limit active. Skipping email. Time since last: {current_time - self.last_email_sent_time:.1f}s")
            return

        # Check severity threshold
        severity = alert_data.get('severity', 'low')
        if severity:
            severity = severity.lower()
        else:
            severity = 'low'
            
        allowed_severities = email_config.get('alert_severity', ['high', 'critical'])
        logger.info(f"DEBUG: Alert severity: {severity}, allowed: {allowed_severities}")
        
        if severity not in allowed_severities:
            logger.info(f"DEBUG: Severity '{severity}' not in allowed list {allowed_severities}, skipping email")
            return
            
        logger.info(f"DEBUG: Proceeding to send email for {severity} severity threat")
        try:
            self._send_email_notification(alert_data, email_config)
            self.last_email_sent_time = time.time()  # Update last sent time
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")

    def _send_email_notification(self, alert_data: Dict[str, Any], config: Dict[str, Any]):
        """Send email aleart via SMTP"""
        sender = config.get('sender_email', 'noreply@threat-detect.com')
        recipients = config.get('recipients', [])
        password = config.get('sender_password', '')
        smtp_server = config.get('smtp_server', 'localhost')
        smtp_port = config.get('smtp_port', 25)
        
        if not recipients:
            logger.warning("No recipients configured for email alerts")
            return
            
        if "${" in password or not password:
            logger.warning("Email password not configured. Skipping email send.")
            logger.info(f"WOULD SEND EMAIL TO: {recipients}")
            return

        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = ", ".join(recipients)
        msg['Subject'] = f"[{alert_data.get('severity', 'UNKNOWN').upper()}] Threat Detected: {alert_data.get('threat_class')}"
        
        body = f"""
        THREAT DETECTION ALERT
        ======================
        
        Severity: {alert_data.get('severity', 'UNKNOWN').upper()}
        Type: {alert_data.get('threat_class', 'Unknown')}
        Time: {alert_data.get('timestamp')}
        
        Source: {alert_data.get('source_ip')}
        Target: {alert_data.get('destination')}
        
        Description:
        {alert_data.get('description')}
        
        Action Taken: Alert Sent
        
        --
        ML Threat Detection System
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Connect to SMTP Server
        try:
            server = smtplib.SMTP(smtp_server, smtp_port)
            if config.get('use_tls', True):
                server.starttls()
            
            if password:
                server.login(sender, password)
                
            server.send_message(msg)
            server.quit()
            logger.info(f"Email alert sent to {len(recipients)} recipients")
        except Exception as e:
            logger.error(f"SMTP Error: {e}")
            raise
    
    def _isolate_network_segment(self, ip_address: str) -> Dict[str, Any]:
        """Isolate network segment (simulated)"""
        # In production: Use SDN API, firewall API, etc.
        logger.info(f"Isolating network segment for {ip_address}")
        return {'isolated': True, 'vlan': 'quarantine'}
    
    def _add_to_blocklist(self, ip_address: str) -> Dict[str, Any]:
        """Add IP to blocklist (simulated)"""
        # In production: Update firewall, IPS, WAF rules
        logger.info(f"Adding {ip_address} to blocklist")
        return {'blocked': True, 'rule_id': 'auto_block_001'}
    
    def _collect_forensic_evidence(self, threat_details: Dict[str, Any]):
        """Collect forensic evidence"""
        # In production: Capture memory dumps, network traffic, logs
        logger.info("Collecting forensic evidence")
    
    def _log_action(
        self,
        threat_id: str,
        action: ResponseAction,
        risk_assessment: Dict[str, Any],
        result: Dict[str, Any]
    ):
        """Log response action for audit trail"""
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'threat_id': threat_id,
            'action': action.value,
            'risk_score': risk_assessment['risk_score'],
            'severity': risk_assessment['severity'],
            'result': result,
            'automated': self.automation_enabled
        }
        
        self.action_history.append(log_entry)
        
        # Write to audit log
        logger.info(f"ACTION LOG: {json.dumps(log_entry)}")
    
    def get_action_history(
        self,
        limit: int = 100,
        severity: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve action history"""
        
        history = self.action_history[-limit:]
        
        if severity:
            history = [
                entry for entry in history 
                if entry.get('severity') == severity
            ]
        
        return history


class AttackChainReconstructor:
    """Reconstruct attack chains from multiple events"""
    
    def __init__(self):
        self.attack_chains: Dict[str, List[Dict[str, Any]]] = {}
        
    def add_event(self, event: Dict[str, Any]):
        """Add event to attack chain reconstruction"""
        
        # Group events by source IP or user
        chain_key = event.get('src_ip') or event.get('user_id', 'unknown')
        
        if chain_key not in self.attack_chains:
            self.attack_chains[chain_key] = []
        
        self.attack_chains[chain_key].append(event)
        
    def reconstruct_chain(self, chain_key: str) -> Dict[str, Any]:
        """Reconstruct full attack chain"""
        
        if chain_key not in self.attack_chains:
            return {'chain': [], 'stages': []}
        
        events = sorted(
            self.attack_chains[chain_key],
            key=lambda x: x.get('timestamp', '')
        )
        
        # Identify attack stages
        stages = self._identify_attack_stages(events)
        
        return {
            'chain_key': chain_key,
            'events': events,
            'stages': stages,
            'duration': self._calculate_duration(events),
            'severity': self._assess_chain_severity(stages)
        }
    
    def _identify_attack_stages(
        self,
        events: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify stages of cyber kill chain"""
        
        stages = []
        threat_classes = [e.get('threat_class') for e in events]
        
        # Map threat classes to kill chain stages
        if 'port_scan' in threat_classes:
            stages.append('reconnaissance')
        if 'brute_force' in threat_classes:
            stages.append('initial_access')
        if 'privilege_escalation' in threat_classes:
            stages.append('privilege_escalation')
        if 'lateral_movement' in threat_classes:
            stages.append('lateral_movement')
        if 'data_exfiltration' in threat_classes:
            stages.append('exfiltration')
        
        return stages
    
    def _calculate_duration(self, events: List[Dict[str, Any]]) -> float:
        """Calculate attack duration in seconds"""
        if len(events) < 2:
            return 0.0
        
        first = datetime.fromisoformat(events[0]['timestamp'])
        last = datetime.fromisoformat(events[-1]['timestamp'])
        
        return (last - first).total_seconds()
    
    def _assess_chain_severity(self, stages: List[str]) -> str:
        """Assess overall attack chain severity"""
        
        if len(stages) >= 4:
            return 'critical'
        elif len(stages) >= 2:
            return 'high'
        elif len(stages) >= 1:
            return 'medium'
        else:
            return 'low'


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    config = {
        'automation': {
            'enabled': True,
            'require_approval_threshold': 80
        },
        'actions': {
            'monitor': {'risk_score_range': [0, 40], 'action': 'monitor'},
            'alert': {'risk_score_range': [41, 60], 'action': 'alert'},
            'isolate': {'risk_score_range': [61, 80], 'action': 'isolate'},
            'block': {'risk_score_range': [81, 100], 'action': 'block'}
        }
    }
    
    engine = ResponseEngine(config)
    
    # Simulate threat detection
    threat_details = {
        'threat_class': 'ransomware',
        'src_ip': '10.0.0.100',
        'dst_ip': '192.168.1.50',
        'severity': 'critical'
    }
    
    # Calculate risk
    risk_assessment = engine.calculate_risk_score(
        anomaly_score=0.95,
        classification_confidence=0.9,
        threat_class='ransomware',
        asset_criticality=9.0,
        user_risk_score=3.0
    )
    
    print(f"Risk Assessment: {json.dumps(risk_assessment, indent=2)}")
    
    # Execute response
    response = engine.execute_response(
        threat_id='THREAT-001',
        risk_assessment=risk_assessment,
        threat_details=threat_details
    )
    
    print(f"\nResponse: {json.dumps(response, indent=2)}")
