import { store } from '../store/store';
import { setConnectionStatus, setSystemMessage } from '../store/slices/systemSlice';
import { updateAccount } from '../store/slices/accountSlice';
import { updatePositions } from '../store/slices/positionsSlice';
import { addLog } from '../store/slices/logSlice';

const WEBSOCKET_URL = "ws://localhost:8000/ws";

class WebSocketService {
  private socket: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectInterval = 5000; // 5 seconds

  public connect() {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      console.log("WebSocket is already connected.");
      return;
    }

    store.dispatch(setConnectionStatus('connecting'));
    this.socket = new WebSocket(WEBSOCKET_URL);

    this.socket.onopen = () => {
      console.log("WebSocket connection established.");
      store.dispatch(setConnectionStatus('connected'));
      store.dispatch(addLog({ level: 'SYSTEM', message: 'Successfully connected to backend WebSocket.' }));
      this.reconnectAttempts = 0; // Reset on successful connection
    };

    this.socket.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        this.handleMessage(message);
      } catch (error) {
        console.error("Failed to parse incoming message:", event.data);
        store.dispatch(addLog({ level: 'ERROR', message: 'Received unparseable message from backend.' }));
      }
    };

    this.socket.onerror = (error) => {
      console.error("WebSocket error:", error);
      store.dispatch(setConnectionStatus('error'));
      store.dispatch(addLog({ level: 'ERROR', message: 'WebSocket connection error.' }));
    };

    this.socket.onclose = () => {
      console.log("WebSocket connection closed.");
      store.dispatch(setConnectionStatus('disconnected'));
      store.dispatch(addLog({ level: 'SYSTEM', message: 'WebSocket connection closed.' }));
      this.handleReconnect();
    };
  }

  private handleMessage(message: any) {
    if (!message.type || !message.data) {
      console.warn("Received malformed message:", message);
      return;
    }

    switch (message.type) {
      case 'account_update':
        store.dispatch(updateAccount(message.data));
        break;
      case 'positions_update':
        store.dispatch(updatePositions(message.data));
        break;
      case 'log_stream':
        store.dispatch(addLog(message.data));
        break;
      case 'system':
        store.dispatch(setSystemMessage(message.data));
        break;
      case 'echo':
        // You can ignore echos or use them for latency checks
        break;
      default:
        console.warn(`Unknown message type received: ${message.type}`);
    }
  }

  private handleReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`Attempting to reconnect in ${this.reconnectInterval / 1000}s... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
      setTimeout(() => {
        this.connect();
      }, this.reconnectInterval);
    } else {
      console.error("Max reconnect attempts reached. Please check the backend server and refresh the page.");
      store.dispatch(addLog({ level: 'ERROR', message: 'Could not reconnect to backend. Max attempts reached.' }));
    }
  }

  public send(message: any) {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(message));
    } else {
      console.error("Cannot send message, WebSocket is not connected.");
    }
  }
}

const webSocketService = new WebSocketService();
export default webSocketService;
