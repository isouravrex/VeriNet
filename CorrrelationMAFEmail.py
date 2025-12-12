import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

# ============================================================
# EMAIL LIBRARIES
# ============================================================
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

# ============================================================
# MICROSOFT AGENT FRAMEWORK IMPORTS
# ============================================================
from agent_framework import (
    ChatAgent,
    ai_function,
    AgentRunUpdateEvent,
    WorkflowOutputEvent,
    SequentialBuilder,
)
from agent_framework.openai import OpenAIChatClient
from openai import AsyncOpenAI
import httpx
import ssl
import certifi

try:
    import truststore
    HAS_TRUSTSTORE = True
except ImportError:
    HAS_TRUSTSTORE = False

# Load environment variables
load_dotenv()

# Configuration
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
CEREBRAS_BASE_URL = os.getenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
CEREBRAS_MODEL = "llama-3.3-70b"

# EMAIL CONFIGURATION
EMAIL_SENDER = "slope.expedition@gmail.com"
EMAIL_RECIPIENT = "incidentexpedition@gmail.com"
EMAIL_PASSWORD = os.getenv("EMAIL_APP_PASSWORD") 

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ============================================================
# TOOL FUNCTIONS
# ============================================================
class NetworkTools:
    """Network monitoring and incident correlation tools"""

    def __init__(self):
        self.data = {}
        self.correlation_log = []
        self.last_chart_path = None 

        self.sd_wan_file = "SD-WANRouterTelemetry.csv"
        self.wifi_file = "WiFiAccessPoints.csv"
        self.teams_file = "MicrosoftTeamsTelemetry.csv"
        self.endpoint_file = "WindowsEndpointLogs.csv"
        self.zscaler_file = "Z-ScalerProxyLogs.csv"
        self.mtr_file = "MFTeamsRoomsTelemetry.csv"

    def _load_data(self) -> Dict[str, pd.DataFrame]:
        try:
            self.data = {
                "sd_wan_df": pd.read_csv(self.sd_wan_file),
                "wifi_df": pd.read_csv(self.wifi_file),
                "teams_df": pd.read_csv(self.teams_file),
                "endpoint_df": pd.read_csv(self.endpoint_file),
                "zscaler_df": pd.read_csv(self.zscaler_file),
                "mtr_df": pd.read_csv(self.mtr_file),
            }
            logging.info("✅ All telemetry data loaded successfully")
            return self.data
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return {}

    @ai_function(description="Retrieve historical incidents for a user from the incident database.")
    def get_user_incidents(self, user_id: str) -> str:
        result = {
            "user_id": user_id,
            "incident_count": 3,
            "last_incident": "2025-12-09",
            "common_issues": ["network_latency", "packet_loss"],
            "avg_resolution_time": "2.5 hours",
        }
        return json.dumps(result, indent=2)

    @ai_function(description="Get current health status and metrics for a network location.")
    def get_location_health(self, location: str) -> str:
        result = {
            "location": location,
            "status": "degraded",
            "active_issues": 2,
            "bandwidth_utilization": "78%",
            "last_check": datetime.utcnow().isoformat(),
            "router_status": "operational",
        }
        return json.dumps(result, indent=2)

    @ai_function(description="Analyze network metrics and detect anomalies with severity assessment.")
    def analyze_metrics(self, packet_loss: float, jitter: float, latency: float = 0.0) -> str:
        anomalies = (packet_loss > 5.0) or (jitter > 100.0)
        if packet_loss > 15.0 or jitter > 300.0:
            severity = "Critical"
        elif packet_loss > 10.0 or jitter > 200.0:
            severity = "High"
        elif anomalies:
            severity = "Medium"
        else:
            severity = "Low"

        result = {
            "anomalies_detected": anomalies,
            "severity": severity,
            "packet_loss_status": "high" if packet_loss > 5.0 else "normal",
            "jitter_status": "high" if jitter > 100.0 else "normal",
            "recommendation": "Investigate router health" if anomalies else "No action needed",
        }
        return json.dumps(result, indent=2)

    @ai_function(description="Retrieve Microsoft Teams call quality metrics for a user.")
    def get_teams_call_quality(self, user_id: str) -> str:
        result = {
            "user_id": user_id,
            "recent_calls": 5,
            "avg_mos_score": 3.2,
            "poor_quality_calls": 2,
            "common_issues": ["high_jitter", "packet_loss"],
            "last_call_status": "poor",
        }
        return json.dumps(result, indent=2)

    @ai_function(description="Create an incident ticket in the ticketing system.")
    def create_ticket(self, severity: str, description: str, assignee: str = "Network Operations") -> str:
        ticket_id = f"INC-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        result = {
            "ticket_id": ticket_id,
            "severity": severity,
            "description": description,
            "status": "created",
            "assignee": assignee,
            "created_at": datetime.utcnow().isoformat(),
        }
        logging.info(f"🎫 TICKET CREATED: {ticket_id}")
        return json.dumps(result, indent=2)

    # ============================================================
    # EMAIL FUNCTION
    # ============================================================
    def send_email_report(self, subject: str, html_content: str) -> str:
        """Sends an email using the pre-configured Gmail credentials."""
        logging.info("📧 EMAIL TOOL INVOKED...") 
        
        sender = EMAIL_SENDER
        recipient = EMAIL_RECIPIENT
        raw_password = EMAIL_PASSWORD
        
        if raw_password:
            password = raw_password.replace(" ", "")
        else:
            password = None

        if not password:
            logging.error("❌ Email failed: EMAIL_APP_PASSWORD is not set.")
            return json.dumps({"status": "error", "message": "Email password not configured."})
        
        try:
            msg = MIMEMultipart("related")
            msg["Subject"] = subject
            msg["From"] = sender
            msg["To"] = recipient

            msg_alternative = MIMEMultipart("alternative")
            msg.attach(msg_alternative)
            msg_text = MIMEText(html_content, "html")
            msg_alternative.attach(msg_text)

            if self.last_chart_path and os.path.exists(self.last_chart_path):
                logging.info(f"📎 Found chart to attach: {self.last_chart_path}")
                with open(self.last_chart_path, "rb") as f:
                    img_data = f.read()
                
                img = MIMEImage(img_data)
                img.add_header("Content-ID", "<analysis_chart>")
                img.add_header("Content-Disposition", "inline", filename=os.path.basename(self.last_chart_path))
                msg.attach(img)
            else:
                logging.warning("⚠️ No chart image found to attach. Sending text only.")

            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(sender, password)
                server.sendmail(sender, recipient, msg.as_string())
            
            logging.info(f"✅ EMAIL SENT SUCCESSFULLY to {recipient}")
            return json.dumps({"status": "success", "message": "Email sent successfully."})

        except Exception as e:
            logging.error(f"❌ Email sending failed: {str(e)}")
            return json.dumps({"status": "error", "message": str(e)})

    # ============================================================
    # CORE LOGIC
    # ============================================================
    def correlate_telemetry(self, target_user_email: str, target_user_id: str, target_location: str, loss_threshold: float = 5.0, jitter_threshold: float = 100.0) -> str:
        if not self.data: self._load_data()
        if not self.data: return json.dumps({"error": "Failed to load telemetry data"})

        # Normalization
        for key, df in self.data.items():
            df.columns = df.columns.str.strip()
            if "Timestamp" in df.columns:
                df["Timestamp"] = pd.to_datetime(df["Timestamp"])
                df.sort_values("Timestamp", inplace=True)

        time_tolerance = "1min"
        sd_wan_target = self.data["sd_wan_df"][self.data["sd_wan_df"]["Router_ID"].str.contains(target_location, na=False)].copy()

        if target_user_email and target_user_id:
            teams_target = self.data["teams_df"][(self.data["teams_df"]["User_Principal_Name"] == target_user_email) & (self.data["teams_df"]["User_ID"] == target_user_id)].copy()
        elif target_user_email:
            teams_target = self.data["teams_df"][self.data["teams_df"]["User_Principal_Name"] == target_user_email].copy()
        else:
            teams_target = self.data["teams_df"][self.data["teams_df"]["User_ID"] == target_user_id].copy()

        if teams_target.empty:
            return json.dumps({"error": "No Teams data found for user", "user": target_user_email})

        master_df = pd.merge_asof(
            teams_target,
            sd_wan_target[["Timestamp", "Packet_Loss_%", "Jitter_ms", "Status_Code"]],
            on="Timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(time_tolerance),
            suffixes=("_Teams", "_SDWAN"),
        )

        if "Packet_Loss_%_SDWAN" in master_df.columns and "Net_Jitter_ms" in master_df.columns:
            correlation_window = master_df[(master_df["Packet_Loss_%_SDWAN"] > loss_threshold) & (master_df["Net_Jitter_ms"] > jitter_threshold)]
        else:
            correlation_window = pd.DataFrame()

        if correlation_window.empty:
            result = {"status": "no_incident", "message": "No strong correlation found"}
        else:
            start = correlation_window["Timestamp"].min()
            end = correlation_window["Timestamp"].max()
            avg_loss = correlation_window["Packet_Loss_%_SDWAN"].mean()
            max_loss = correlation_window["Packet_Loss_%_SDWAN"].max()
            avg_jitter = correlation_window["Net_Jitter_ms"].mean()
            max_jitter = correlation_window["Net_Jitter_ms"].max()

            self.correlation_log.append({
                "user": target_user_email or target_user_id,
                "location": target_location,
                "start": str(start),
                "end": str(end),
                "master_df": master_df,
                "correlation_window": correlation_window,
                "sd_wan_target": sd_wan_target,
            })

            result = {
                "status": "incident_detected",
                "user": target_user_email or target_user_id,
                "location": target_location,
                "metrics": {
                    "avg_packet_loss": round(avg_loss, 2),
                    "max_packet_loss": round(max_loss, 2),
                    "avg_jitter": round(avg_jitter, 2),
                    "max_jitter": round(max_jitter, 2),
                },
            }
        return json.dumps(result, indent=2, default=str)

    def visualize_correlation(self, target_user: str, target_location: str, loss_threshold: float, jitter_threshold: float, show_plot: bool = True):
        if not self.correlation_log: return

        latest = None
        for log in reversed(self.correlation_log):
            if log["user"] == target_user and log["location"] == target_location:
                latest = log
                break

        if not latest: return

        plt.figure(figsize=(15, 12))
        sd_wan_target = latest["sd_wan_target"]
        master_df = latest["master_df"]

        plt.subplot(3, 1, 1)
        if "Packet_Loss_%" in sd_wan_target.columns:
            plt.plot(sd_wan_target["Timestamp"], sd_wan_target["Packet_Loss_%"], label="Packet Loss %", color="red", marker="o")
        if "Jitter_ms" in sd_wan_target.columns:
            plt.plot(sd_wan_target["Timestamp"], sd_wan_target["Jitter_ms"], label="Jitter (ms)", color="orange", linestyle="--")
        plt.axhline(y=loss_threshold, color="red", linestyle=":", label=f"Threshold ({loss_threshold}%)")
        plt.title(f"Network Layer: SD-WAN ({target_location})")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(3, 1, 2)
        if not master_df.empty and "Net_Jitter_ms" in master_df.columns:
            plt.plot(master_df["Timestamp"], master_df["Net_Jitter_ms"], label="Teams Jitter", color="blue", marker="x")
        plt.axhline(y=jitter_threshold, color="blue", linestyle=":", label=f"Threshold ({jitter_threshold}ms)")
        plt.title(f"Application Layer ({target_user})")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(3, 1, 3)
        correlation_window = latest["correlation_window"]
        if not correlation_window.empty:
            plt.scatter(correlation_window["Timestamp"], correlation_window["Packet_Loss_%_SDWAN"], c="red", s=100, label="Correlated Incident")
        plt.title("Correlation Window")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        
        safe_user = target_user.replace("@", "_").replace(".", "_")
        image_file = f"maf_analysis_{safe_user}_{target_location}.png"
        plt.savefig(image_file)
        self.last_chart_path = os.path.abspath(image_file)
        logging.info(f"📊 Visualization saved to: {self.last_chart_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

# Initialize Tools
network_tools = NetworkTools()


# ============================================================
# SSL / CLIENT SETUP
# ============================================================
def _build_ssl_context() -> ssl.SSLContext | bool:
    insecure = os.getenv("INSECURE_SKIP_VERIFY", "").lower() in ("1", "true", "yes")
    if insecure: return False
    ca_file = os.getenv("SSL_CERT_FILE") or os.getenv("REQUESTS_CA_BUNDLE")
    if ca_file and os.path.exists(ca_file): return ssl.create_default_context(cafile=ca_file)
    if HAS_TRUSTSTORE: return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return ssl.create_default_context(cafile=certifi.where())

def create_cerebras_client() -> OpenAIChatClient:
    httpx_client = httpx.AsyncClient(verify=_build_ssl_context(), timeout=httpx.Timeout(600.0, connect=10.0), trust_env=True)
    oa_client = AsyncOpenAI(api_key=CEREBRAS_API_KEY, base_url=CEREBRAS_BASE_URL, http_client=httpx_client)
    return OpenAIChatClient(async_client=oa_client, model_id=CEREBRAS_MODEL)


# ============================================================
# CREATE AGENTS
# ============================================================
async def create_network_analysis_agents():
    client = create_cerebras_client()

    network_analyzer = ChatAgent(
        name="NetworkAnalyzer",
        instructions="You are ONLY the NetworkAnalyzer. You analyze packet loss and jitter. Do NOT speak for other agents. Do NOT execute tools for other agents (like create_ticket).",
        chat_client=client,
        tools=[network_tools.get_location_health, network_tools.analyze_metrics],
    )

    application_analyzer = ChatAgent(
        name="ApplicationAnalyzer",
        instructions="You are ONLY the ApplicationAnalyzer. You analyze Teams metrics. Do NOT speak for other agents. Do NOT execute tools for other agents.",
        chat_client=client,
        tools=[network_tools.get_user_incidents, network_tools.get_teams_call_quality],
    )

    incident_coordinator = ChatAgent(
        name="IncidentCoordinator",
        instructions="You are ONLY the IncidentCoordinator. Synthesize findings. YOU MUST CALL 'create_ticket'. Do not speak for other agents.",
        chat_client=client,
        tools=[network_tools.create_ticket],
    )

    logging.info("✅ Created 3 ChatAgents.")
    return network_analyzer, application_analyzer, incident_coordinator


# ============================================================
# HTML GENERATOR (UPDATED TO INCLUDE AGENT TEXT)
# ============================================================
def generate_python_html_report(user, location, metrics, agent_insights):
    """Generates the HTML string with AI Analysis."""
    avg_loss = metrics.get('avg_packet_loss', 'N/A')
    avg_jitter = metrics.get('avg_jitter', 'N/A')
    
    # Format Agent Insights into HTML
    insights_html = ""
    for agent_name, text in agent_insights.items():
        if text.strip():
            insights_html += f"""
            <div style="margin-bottom: 15px; border-left: 4px solid #0078D4; padding-left: 10px;">
                <strong>{agent_name}:</strong>
                <p style="margin-top: 5px;">{text}</p>
            </div>
            """

    return f"""
    <html>
    <body style="font-family: Arial, sans-serif;">
        <div style="background-color: #0078D4; color: white; padding: 15px;">
            <h2>INCIDENT REPORT: {location}</h2>
        </div>
        <div style="padding: 20px;">
            <p><strong>User:</strong> {user}</p>
            <p><strong>Location:</strong> {location}</p>
            
            <h3>Detected Metrics:</h3>
            <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; width: 50%;">
                <tr style="background-color: #f2f2f2;"><th>Metric</th><th>Value</th></tr>
                <tr><td>Avg Packet Loss</td><td>{avg_loss}%</td></tr>
                <tr><td>Avg Jitter</td><td>{avg_jitter}ms</td></tr>
            </table>

            <h3>🤖 Agent AI Analysis:</h3>
            {insights_html}

            <br>
            <p><strong>Status:</strong> Ticket Created.</p>
            <p><strong>Visual Analysis:</strong> See attached chart.</p>
            <br>
            <img src='cid:analysis_chart' style='max-width:100%; border:1px solid #ddd;'/>
        </div>
    </body>
    </html>
    """

# ============================================================
# WORKFLOW EXECUTION
# ============================================================
async def NetworkIncidentCorrelation(
    target_user_email: str = "sarah.j@contoso.com",
    target_user_id: str = "U-1099",
    target_location: str = "DAL",
    loss_threshold: float = 5.0,
    jitter_threshold: float = 100.0,
):
    print("\n" + "=" * 100)
    print("🚀 NETWORK INCIDENT CORRELATION & EMAIL REPORTING")
    print("=" * 100)

    # 1. Data Correlation
    print("\n📊 PHASE 1: DATA CORRELATION")
    correlation_result = network_tools.correlate_telemetry(
        target_user_email=target_user_email,
        target_user_id=target_user_id,
        target_location=target_location,
        loss_threshold=loss_threshold,
        jitter_threshold=jitter_threshold,
    )
    correlation_data = json.loads(correlation_result)
    print(f"Status: {correlation_data.get('status')}")

    # Generate Chart if incident detected
    if correlation_data.get("status") == "incident_detected":
        print("\n📸 Pre-generating analysis chart for email attachment...")
        network_tools.visualize_correlation(
            target_user=target_user_email,
            target_location=target_location,
            loss_threshold=loss_threshold,
            jitter_threshold=jitter_threshold,
            show_plot=False 
        )

    # 2. Create Agents
    network_analyzer, application_analyzer, incident_coordinator = await create_network_analysis_agents()

    # 3. Build Workflow
    workflow = (
        SequentialBuilder()
        .participants([network_analyzer, application_analyzer, incident_coordinator])
        .build()
    )

    # 4. Run Workflow and CAPTURE TEXT
    print("\n▶️ PHASE 3: EXECUTING MULTI-AGENT ANALYSIS")
    
    incident_prompt = f"""
    AUTOMATED INCIDENT PIPELINE
    ===========================
    User: {target_user_email}
    Location: {target_location}
    Correlation Data: {json.dumps(correlation_data)}
    
    INSTRUCTIONS:
    1. NetworkAnalyzer: Briefly analyze network metrics. STOP after analysis.
    2. ApplicationAnalyzer: Briefly analyze app metrics. STOP after analysis.
    3. IncidentCoordinator: Summarize and CALL 'create_ticket(severity="High", description="High Packet Loss")'. STOP.
    """

    agent_outputs = {} # Dictionary to store what each agent says
    last_executor_id = None

    try:
        async for event in workflow.run_stream(incident_prompt):
            if isinstance(event, AgentRunUpdateEvent):
                # Print to console
                if hasattr(event, "executor_id") and event.executor_id != last_executor_id:
                    print(f"\n🤖 [{event.executor_id}]:", end=" ", flush=True)
                    last_executor_id = event.executor_id
                if hasattr(event, "data"):
                    print(event.data, end="", flush=True)
                    
                    # FIX: Safely handle content concatenation
                    if event.executor_id:
                        current_text = agent_outputs.get(event.executor_id, "")
                        # Force conversion to string to prevent TypeError
                        new_content = str(event.data)
                        agent_outputs[event.executor_id] = current_text + new_content

            elif isinstance(event, WorkflowOutputEvent):
                print("\n\n✅ WORKFLOW COMPLETED")
    except Exception as ex:
        logging.error(f"Workflow failed: {ex}")

    # ============================================================
    # PHASE 4: DETERMINISTIC EMAIL DELIVERY (With Captured Text)
    # ============================================================
    if correlation_data.get("status") == "incident_detected":
        print("\n" + "=" * 100)
        print("📧 PHASE 4: EXECUTING EMAIL REPORTING (Robust Mode)")
        print("=" * 100)
        
        # Generate HTML combining Metrics + Captured Agent Text
        html_payload = generate_python_html_report(
            target_user_email, 
            target_location, 
            correlation_data.get("metrics", {}),
            agent_outputs
        )
        
        print(f"\n🤖 [EmailReporter]: Initiating secure email transfer to {EMAIL_RECIPIENT}...")
        
        # Call the tool directly
        email_result = network_tools.send_email_report(
            subject=f"INCIDENT ALERT: {target_location}",
            html_content=html_payload
        )
        print(f"   Status: {email_result}")

    # 5. Final Visualization (Popup)
    print("\n" + "=" * 100)
    print("📊 PHASE 5: SHOWING VISUALIZATION (User Window)")
    print("=" * 100)
    if network_tools.correlation_log:
        network_tools.visualize_correlation(
            target_user=target_user_email,
            target_location=target_location,
            loss_threshold=loss_threshold,
            jitter_threshold=jitter_threshold,
            show_plot=True 
        )

if __name__ == "__main__":
    asyncio.run(
        NetworkIncidentCorrelation(
            target_user_email="sarah.j@contoso.com",
            target_location="DAL"
        )
    )