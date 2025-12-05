"""
AlertManager - 專業級告警系統
支持多渠道通知 (Telegram, Email, Webhook)
"""
import logging
import asyncio
import aiohttp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import os

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """告警級別"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """告警渠道"""
    TELEGRAM = "telegram"
    EMAIL = "email"
    WEBHOOK = "webhook"
    CONSOLE = "console"
    FILE = "file"


@dataclass
class AlertConfig:
    """告警配置"""
    # Telegram配置
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    
    # Email配置
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    email_recipients: List[str] = field(default_factory=list)
    
    # Webhook配置
    webhook_urls: List[str] = field(default_factory=list)
    
    # 通用配置
    min_alert_level: AlertLevel = AlertLevel.WARNING
    rate_limit_seconds: int = 60  # 同類告警間隔
    log_file_path: Optional[Path] = None
    
    @classmethod
    def from_env(cls) -> "AlertConfig":
        """從環境變數載入配置"""
        return cls(
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
            smtp_server=os.getenv("SMTP_SERVER", "smtp.gmail.com"),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            smtp_username=os.getenv("SMTP_USERNAME"),
            smtp_password=os.getenv("SMTP_PASSWORD"),
            email_recipients=os.getenv("ALERT_EMAIL_RECIPIENTS", "").split(","),
            webhook_urls=os.getenv("ALERT_WEBHOOK_URLS", "").split(","),
        )


@dataclass
class Alert:
    """告警消息"""
    level: AlertLevel
    title: str
    message: str
    source: str = "TradingSystem"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    def format_telegram(self) -> str:
        """格式化為Telegram消息"""
        emoji = {
            AlertLevel.DEBUG: "🔍",
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨",
        }
        
        text = f"{emoji.get(self.level, '📢')} *{self.title}*\n\n"
        text += f"📍 Source: `{self.source}`\n"
        text += f"⏰ Time: `{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}`\n\n"
        text += f"{self.message}"
        
        if self.metadata:
            text += "\n\n📊 *Details:*\n"
            for key, value in self.metadata.items():
                text += f"• {key}: `{value}`\n"
        
        return text
    
    def format_email_html(self) -> str:
        """格式化為HTML郵件"""
        colors = {
            AlertLevel.DEBUG: "#6c757d",
            AlertLevel.INFO: "#17a2b8",
            AlertLevel.WARNING: "#ffc107",
            AlertLevel.ERROR: "#dc3545",
            AlertLevel.CRITICAL: "#721c24",
        }
        
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <div style="background-color: {colors.get(self.level, '#333')}; color: white; padding: 15px; border-radius: 5px;">
                <h2>{self.title}</h2>
            </div>
            <div style="padding: 20px;">
                <p><strong>Source:</strong> {self.source}</p>
                <p><strong>Time:</strong> {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Level:</strong> {self.level.value.upper()}</p>
                <hr>
                <p>{self.message}</p>
        """
        
        if self.metadata:
            html += "<h4>Details:</h4><ul>"
            for key, value in self.metadata.items():
                html += f"<li><strong>{key}:</strong> {value}</li>"
            html += "</ul>"
        
        html += """
            </div>
        </body>
        </html>
        """
        return html


class AlertManager:
    """
    專業級告警管理器
    
    功能：
    1. 多渠道通知 (Telegram, Email, Webhook)
    2. 告警級別過濾
    3. 速率限制 (防止告警風暴)
    4. 異步發送
    5. 告警歷史記錄
    """
    
    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig.from_env()
        self._last_alerts: Dict[str, datetime] = {}  # 速率限制
        self._alert_history: List[Alert] = []
        self._handlers: Dict[AlertChannel, Callable] = {
            AlertChannel.TELEGRAM: self._send_telegram,
            AlertChannel.EMAIL: self._send_email,
            AlertChannel.WEBHOOK: self._send_webhook,
            AlertChannel.CONSOLE: self._send_console,
            AlertChannel.FILE: self._send_file,
        }
    
    async def send_alert(
        self,
        alert: Alert,
        channels: Optional[List[AlertChannel]] = None
    ) -> Dict[AlertChannel, bool]:
        """
        發送告警
        
        Args:
            alert: 告警對象
            channels: 發送渠道列表，None則自動選擇
            
        Returns:
            每個渠道的發送結果
        """
        # 級別過濾
        if alert.level.value < self.config.min_alert_level.value:
            return {}
        
        # 速率限制
        alert_key = f"{alert.source}:{alert.title}"
        if not self._check_rate_limit(alert_key):
            logger.debug(f"Alert rate limited: {alert_key}")
            return {}
        
        # 記錄告警
        self._alert_history.append(alert)
        self._last_alerts[alert_key] = datetime.now()
        
        # 選擇渠道
        if channels is None:
            channels = self._auto_select_channels(alert.level)
        
        # 並行發送
        results = {}
        tasks = []
        
        for channel in channels:
            handler = self._handlers.get(channel)
            if handler:
                tasks.append((channel, handler(alert)))
        
        for channel, task in tasks:
            try:
                result = await task
                results[channel] = result
            except Exception as e:
                logger.error(f"Failed to send alert via {channel}: {e}")
                results[channel] = False
        
        return results
    
    def send_alert_sync(
        self,
        alert: Alert,
        channels: Optional[List[AlertChannel]] = None
    ) -> Dict[AlertChannel, bool]:
        """同步發送告警"""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.send_alert(alert, channels))
        finally:
            loop.close()
    
    def _check_rate_limit(self, alert_key: str) -> bool:
        """檢查速率限制"""
        last_time = self._last_alerts.get(alert_key)
        if last_time is None:
            return True
        
        elapsed = (datetime.now() - last_time).total_seconds()
        return elapsed >= self.config.rate_limit_seconds
    
    def _auto_select_channels(self, level: AlertLevel) -> List[AlertChannel]:
        """自動選擇發送渠道"""
        channels = [AlertChannel.CONSOLE]
        
        if self.config.log_file_path:
            channels.append(AlertChannel.FILE)
        
        if level in [AlertLevel.WARNING, AlertLevel.ERROR, AlertLevel.CRITICAL]:
            if self.config.telegram_bot_token:
                channels.append(AlertChannel.TELEGRAM)
            if self.config.webhook_urls:
                channels.append(AlertChannel.WEBHOOK)
        
        if level == AlertLevel.CRITICAL:
            if self.config.smtp_username:
                channels.append(AlertChannel.EMAIL)
        
        return channels
    
    async def _send_telegram(self, alert: Alert) -> bool:
        """發送Telegram消息"""
        if not self.config.telegram_bot_token or not self.config.telegram_chat_id:
            return False
        
        url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
        payload = {
            "chat_id": self.config.telegram_chat_id,
            "text": alert.format_telegram(),
            "parse_mode": "Markdown"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    result = await resp.json()
                    return result.get("ok", False)
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False
    
    async def _send_email(self, alert: Alert) -> bool:
        """發送Email"""
        if not self.config.smtp_username or not self.config.email_recipients:
            return False
        
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[{alert.level.value.upper()}] {alert.title}"
            msg["From"] = self.config.smtp_username
            msg["To"] = ", ".join(self.config.email_recipients)
            
            html_part = MIMEText(alert.format_email_html(), "html")
            msg.attach(html_part)
            
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.smtp_username, self.config.smtp_password)
                server.send_message(msg)
            
            return True
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return False
    
    async def _send_webhook(self, alert: Alert) -> bool:
        """發送Webhook"""
        if not self.config.webhook_urls:
            return False
        
        success = True
        async with aiohttp.ClientSession() as session:
            for url in self.config.webhook_urls:
                if not url.strip():
                    continue
                try:
                    async with session.post(url, json=alert.to_dict()) as resp:
                        if resp.status >= 400:
                            success = False
                except Exception as e:
                    logger.error(f"Webhook send failed to {url}: {e}")
                    success = False
        
        return success
    
    async def _send_console(self, alert: Alert) -> bool:
        """輸出到控制台"""
        level_map = {
            AlertLevel.DEBUG: logging.DEBUG,
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL,
        }
        
        log_level = level_map.get(alert.level, logging.INFO)
        logger.log(log_level, f"[{alert.source}] {alert.title}: {alert.message}")
        return True
    
    async def _send_file(self, alert: Alert) -> bool:
        """寫入日誌文件"""
        if not self.config.log_file_path:
            return False
        
        try:
            with open(self.config.log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert.to_dict()) + "\n")
            return True
        except Exception as e:
            logger.error(f"File write failed: {e}")
            return False
    
    def get_alert_history(
        self,
        level: Optional[AlertLevel] = None,
        limit: int = 100
    ) -> List[Alert]:
        """獲取告警歷史"""
        history = self._alert_history
        
        if level:
            history = [a for a in history if a.level == level]
        
        return history[-limit:]


# 預定義告警工廠函數
def create_trade_alert(
    symbol: str,
    action: str,
    units: float,
    price: float,
    pnl: Optional[float] = None
) -> Alert:
    """創建交易告警"""
    return Alert(
        level=AlertLevel.INFO,
        title=f"Trade Executed: {symbol}",
        message=f"{action} {abs(units)} units at {price}",
        source="OrderManager",
        metadata={
            "symbol": symbol,
            "action": action,
            "units": units,
            "price": price,
            "pnl": pnl
        }
    )


def create_risk_alert(
    risk_type: str,
    current_value: float,
    threshold: float,
    message: str
) -> Alert:
    """創建風險告警"""
    level = AlertLevel.WARNING if current_value < threshold else AlertLevel.CRITICAL
    return Alert(
        level=level,
        title=f"Risk Alert: {risk_type}",
        message=message,
        source="RiskManager",
        metadata={
            "risk_type": risk_type,
            "current_value": current_value,
            "threshold": threshold
        }
    )


def create_system_alert(
    component: str,
    status: str,
    message: str,
    is_error: bool = False
) -> Alert:
    """創建系統告警"""
    return Alert(
        level=AlertLevel.ERROR if is_error else AlertLevel.INFO,
        title=f"System: {component}",
        message=message,
        source=component,
        metadata={"status": status}
    )


# 全局告警管理器實例
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """獲取全局告警管理器"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # 測試
    manager = AlertManager()
    
    alert = Alert(
        level=AlertLevel.WARNING,
        title="Test Alert",
        message="This is a test alert message.",
        source="TestScript",
        metadata={"test_key": "test_value"}
    )
    
    # 同步發送 (僅控制台)
    results = manager.send_alert_sync(alert, [AlertChannel.CONSOLE])
    print(f"Results: {results}")
