<div class="flex flex-col h-[600px] w-full max-w-4xl mx-auto bg-white rounded-lg shadow-xl overflow-hidden border border-gray-200">
  <div class="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
    <div *ngFor="let msg of messages" class="flex" [ngClass]="{'justify-end': msg.role === 'user', 'justify-start': msg.role === 'assistant'}">
      <div class="max-w-[80%] p-3 rounded-lg whitespace-pre-wrap"
           [ngClass]="{'bg-blue-600 text-white': msg.role === 'user', 'bg-white border border-gray-200 text-gray-800 shadow-sm': msg.role === 'assistant'}">
        {{ msg.content }}
      </div>
    </div>
    <div *ngIf="loading" class="flex justify-start">
      <div class="bg-white border border-gray-200 p-3 rounded-lg shadow-sm flex items-center">
        Thinking...
      </div>
    </div>
  </div>

  <div class="p-4 bg-white border-t border-gray-200">
    <div *ngIf="file" class="flex items-center mb-2 p-2 bg-blue-50 text-blue-700 rounded text-sm">
      {{ file.name }}
      <button (click)="file = null" class="ml-auto text-blue-500 hover:text-blue-700">×</button>
    </div>
    <div class="flex gap-2">
      <input type="file" (change)="onFileSelected($event)" class="hidden" #fileInput accept=".pdf,.docx">
      <button (click)="fileInput.click()" class="p-2 text-gray-500 hover:text-blue-600 transition-colors" title="Upload Document">
        📎
      </button>
      <input type="text" [(ngModel)]="input" (keydown.enter)="sendMessage()" placeholder="Ask the agent or upload a document..."
             class="flex-1 p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
      <button (click)="sendMessage()" [disabled]="loading || (!input && !file)"
              class="p-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 transition-colors">
        Send
      </button>
    </div>
  </div>
</div>


import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

interface Message {
    role: 'user' | 'assistant';
    content: string;
}

@Component({
    selector: 'app-chat',
    standalone: true,
    imports: [CommonModule, FormsModule],
    templateUrl: './chat.component.html',
    styleUrls: ['./chat.component.css']
})
export class ChatComponent {
    messages: Message[] = [];
    input: string = '';
    loading: boolean = false;
    file: File | null = null;

    constructor(private http: HttpClient) { }

    onFileSelected(event: any) {
        this.file = event.target.files[0];
    }

    async sendMessage() {
        if (!this.input.trim() && !this.file) return;

        const userMessage = this.input;
        this.messages.push({ role: 'user', content: userMessage + (this.file ? ` [Attached: ${this.file.name}]` : '') });
        this.input = '';
        this.loading = true;

        try {
            let responseText = '';

            if (this.file) {
                const formData = new FormData();
                formData.append('file', this.file);
                try {
                    const uploadRes: any = await this.http.post('http://localhost:8000/ingest', formData).toPromise();
                    responseText += `[System]: ${uploadRes.message}\nPreview: ${uploadRes.content_preview}\n\n`;
                } catch (e) {
                    responseText += `[System]: File upload failed: ${e}\n\n`;
                }
                this.file = null;
            }

            if (userMessage) {
                const res: any = await this.http.post('http://localhost:8000/chat', { message: userMessage }).toPromise();
                responseText += res.response;
            }

            this.messages.push({ role: 'assistant', content: responseText });
        } catch (error) {
            this.messages.push({ role: 'assistant', content: 'Error: Failed to communicate with the agent.' });
        } finally {
            this.loading = false;
        }
    }
}


<div class="admin-container">
  <header class="admin-header">
    <div class="header-content">
      <h1>Admin Panel</h1>
      <a routerLink="/dashboard" class="btn-link">← Back to Dashboard</a>
    </div>
  </header>

  <div class="admin-content">
    <div *ngIf="loading" class="loading">Loading admin data...</div>
    <div *ngIf="error" class="error-message">{{ error }}</div>

    <div *ngIf="!loading && metrics" class="admin-sections">
      <!-- Metrics Section -->
      <section class="metrics-section">
        <h2>System Metrics</h2>
        <div class="metrics-grid">
          <div class="metric-card">
            <div class="metric-value">{{ metrics.totalJobs }}</div>
            <div class="metric-label">Total Jobs</div>
          </div>
          <div class="metric-card success">
            <div class="metric-value">{{ metrics.completedJobs }}</div>
            <div class="metric-label">Completed</div>
          </div>
          <div class="metric-card error">
            <div class="metric-value">{{ metrics.failedJobs }}</div>
            <div class="metric-label">Failed</div>
          </div>
          <div class="metric-card active">
            <div class="metric-value">{{ metrics.activeJobs }}</div>
            <div class="metric-label">Active</div>
          </div>
          <div class="metric-card">
            <div class="metric-value">{{ metrics.errorRate }}%</div>
            <div class="metric-label">Error Rate</div>
          </div>
          <div class="metric-card">
            <div class="metric-value">{{ metrics.averageJobTime }}m</div>
            <div class="metric-label">Avg Job Time</div>
          </div>
        </div>
      </section>

      <!-- Tool Registry Section -->
      <section class="tools-section">
        <h2>Tool Registry & Quotas</h2>
        <div class="tools-list">
          <div *ngFor="let tool of tools" class="tool-card">
            <div class="tool-header">
              <div class="tool-info">
                <h3>{{ tool.name }}</h3>
                <p class="tool-description">{{ tool.description }}</p>
              </div>
              <label class="toggle-switch">
                <input
                  type="checkbox"
                  [checked]="tool.enabled"
                  (change)="toggleTool(tool)"
                />
                <span class="toggle-slider"></span>
              </label>
            </div>

            <div class="tool-quota">
              <div class="quota-header">
                <span>Usage: {{ tool.usage }} / {{ tool.quota }}</span>
                <span class="quota-percentage">{{ getQuotaPercentage(tool) | number:'1.0-0' }}%</span>
              </div>
              <div class="quota-bar">
                <div
                  class="quota-fill"
                  [style.width.%]="getQuotaPercentage(tool)"
                  [style.background-color]="getQuotaColor(tool)"
                ></div>
              </div>
            </div>

            <div class="tool-actions">
              <div class="quota-input-group">
                <label for="quota-{{ tool.id }}">Set Quota:</label>
                <input
                  type="number"
                  id="quota-{{ tool.id }}"
                  [(ngModel)]="tool.quota"
                  min="0"
                  class="quota-input"
                />
                <button
                  class="btn btn-sm btn-primary"
                  (click)="updateToolQuota(tool)"
                >
                  Update
                </button>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  </div>
</div>


.admin-container {
  min-height: 100vh;
  background: #f5f7fa;
}

.admin-header {
  background: white;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  padding: 20px 0;
  margin-bottom: 30px;
}

.header-content {
  max-width: 1400px;
  margin: 0 auto;
  padding: 0 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header-content h1 {
  margin: 0;
  color: #333;
}

.admin-content {
  max-width: 1400px;
  margin: 0 auto;
  padding: 0 20px;
}

.loading {
  text-align: center;
  padding: 40px;
  color: #666;
}

.error-message {
  color: #e74c3c;
  padding: 15px;
  background: #fee;
  border-radius: 6px;
  margin-bottom: 20px;
}

.admin-sections {
  display: flex;
  flex-direction: column;
  gap: 30px;
}

.metrics-section h2,
.tools-section h2 {
  margin: 0 0 20px 0;
  color: #333;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
}

.metric-card {
  background: white;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  text-align: center;
}

.metric-card.success {
  border-left: 4px solid #27ae60;
}

.metric-card.error {
  border-left: 4px solid #e74c3c;
}

.metric-card.active {
  border-left: 4px solid #3498db;
}

.metric-value {
  font-size: 32px;
  font-weight: 600;
  color: #333;
  margin-bottom: 5px;
}

.metric-label {
  font-size: 14px;
  color: #666;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.tools-list {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.tool-card {
  background: white;
  border-radius: 8px;
  padding: 25px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.tool-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 20px;
}

.tool-info {
  flex: 1;
}

.tool-info h3 {
  margin: 0 0 8px 0;
  color: #333;
}

.tool-description {
  margin: 0;
  color: #666;
  font-size: 14px;
}

.toggle-switch {
  position: relative;
  display: inline-block;
  width: 50px;
  height: 24px;
}

.toggle-switch input {
  opacity: 0;
  width: 0;
  height: 0;
}

.toggle-slider {
  position: absolute;
  cursor: pointer;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: #ccc;
  transition: 0.4s;
  border-radius: 24px;
}

.toggle-slider:before {
  position: absolute;
  content: "";
  height: 18px;
  width: 18px;
  left: 3px;
  bottom: 3px;
  background-color: white;
  transition: 0.4s;
  border-radius: 50%;
}

.toggle-switch input:checked + .toggle-slider {
  background-color: #667eea;
}

.toggle-switch input:checked + .toggle-slider:before {
  transform: translateX(26px);
}

.tool-quota {
  margin-bottom: 20px;
}

.quota-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
  font-size: 14px;
  color: #666;
}

.quota-percentage {
  font-weight: 500;
  color: #333;
}

.quota-bar {
  width: 100%;
  height: 12px;
  background: #e9ecef;
  border-radius: 6px;
  overflow: hidden;
}

.quota-fill {
  height: 100%;
  transition: width 0.3s;
  border-radius: 6px;
}

.tool-actions {
  padding-top: 15px;
  border-top: 1px solid #eee;
}

.quota-input-group {
  display: flex;
  align-items: center;
  gap: 10px;
}

.quota-input-group label {
  font-size: 14px;
  color: #555;
  font-weight: 500;
}

.quota-input {
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
  width: 100px;
}

.quota-input:focus {
  outline: none;
  border-color: #667eea;
}

.btn {
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s;
}

.btn-sm {
  padding: 6px 12px;
  font-size: 12px;
}

.btn-primary {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.btn-primary:hover {
  opacity: 0.9;
}

.btn-link {
  color: #667eea;
  text-decoration: none;
  font-size: 14px;
}

.btn-link:hover {
  text-decoration: underline;
}

import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../../services/api.service';
import { AuthService } from '../../../services/auth.service';

interface Tool {
  id: string;
  name: string;
  enabled: boolean;
  quota: number;
  usage: number;
  description: string;
}

interface Metrics {
  totalJobs: number;
  completedJobs: number;
  failedJobs: number;
  activeJobs: number;
  errorRate: number;
  averageJobTime: number;
}

@Component({
  selector: 'app-admin-panel',
  standalone: true,
  imports: [CommonModule, RouterModule, FormsModule],
  templateUrl: './admin-panel.component.html',
  styleUrls: ['./admin-panel.component.css']
})
export class AdminPanelComponent implements OnInit {
  tools: Tool[] = [];
  metrics: Metrics | null = null;
  loading: boolean = true;
  error: string = '';

  constructor(
    private apiService: ApiService,
    private authService: AuthService
  ) {}

  ngOnInit() {
    this.loadData();
  }

  loadData() {
    this.loading = true;
    
    // Load tools and metrics in parallel
    this.apiService.getToolRegistry().subscribe({
      next: (tools) => {
        this.tools = tools;
        this.checkLoadingComplete();
      },
      error: (err) => {
        console.error('Error loading tools:', err);
        // Use mock data if API fails
        this.tools = this.getMockTools();
        this.checkLoadingComplete();
      }
    });

    this.apiService.getAdminMetrics().subscribe({
      next: (metrics) => {
        this.metrics = metrics;
        this.checkLoadingComplete();
      },
      error: (err) => {
        console.error('Error loading metrics:', err);
        // Use mock data if API fails
        this.metrics = this.getMockMetrics();
        this.checkLoadingComplete();
      }
    });
  }

  checkLoadingComplete() {
    if (this.tools.length > 0 && this.metrics) {
      this.loading = false;
    }
  }

  updateToolQuota(tool: Tool) {
    this.apiService.updateToolQuota(tool.id, tool.quota).subscribe({
      next: () => {
        alert(`Quota updated for ${tool.name}`);
      },
      error: (err) => {
        alert(`Failed to update quota: ${err.error?.detail || 'Unknown error'}`);
      }
    });
  }

  toggleTool(tool: Tool) {
    tool.enabled = !tool.enabled;
    // TODO: Implement API call to enable/disable tool
    console.log(`Tool ${tool.id} ${tool.enabled ? 'enabled' : 'disabled'}`);
  }

  getQuotaPercentage(tool: Tool): number {
    if (tool.quota === 0) return 0;
    return Math.min((tool.usage / tool.quota) * 100, 100);
  }

  getQuotaColor(tool: Tool): string {
    const percentage = this.getQuotaPercentage(tool);
    if (percentage >= 90) return '#e74c3c';
    if (percentage >= 70) return '#f39c12';
    return '#27ae60';
  }

  private getMockTools(): Tool[] {
    return [
      {
        id: 'web_search',
        name: 'Web Search',
        enabled: true,
        quota: 100,
        usage: 45,
        description: 'Search the web for relevant information'
      },
      {
        id: 'rag',
        name: 'RAG',
        enabled: true,
        quota: 200,
        usage: 120,
        description: 'Retrieval Augmented Generation from knowledge base'
      },
      {
        id: 'compliance',
        name: 'Compliance Check',
        enabled: true,
        quota: 50,
        usage: 30,
        description: 'PII redaction and compliance verification'
      },
      {
        id: 'citation_validation',
        name: 'Citation Validation',
        enabled: true,
        quota: 100,
        usage: 75,
        description: 'Verify and validate citations in reports'
      }
    ];
  }

  private getMockMetrics(): Metrics {
    return {
      totalJobs: 150,
      completedJobs: 120,
      failedJobs: 10,
      activeJobs: 20,
      errorRate: 6.67,
      averageJobTime: 45.5
    };
  }
}



import logging
import sys
import structlog

def configure_logging():
    """
    Configures structured logging for the application.
    """

    shared_processors = [
        structlog.contextvars.merge_contextvars,  # include contextvars
        structlog.processors.add_log_level,       # log_level=info/debug
        structlog.processors.TimeStamper(fmt="iso"),  # timestamp
    ]

    if sys.stderr.isatty():
        # Pretty color logs for local development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(),
        ]
    else:
        # JSON logs for production / observability dashboards
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Standard library logging → structlog compatibility
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )

logger = structlog.get_logger()


from langfuse import Langfuse
from langfuse.client import LangfuseClient
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import os

# Pull credentials from environment
LANGFUSE_SECRET = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# Initialize global client
langfuse = Langfuse(
    secret_key=LANGFUSE_SECRET,
    public_key=LANGFUSE_PUBLIC,
    host=LANGFUSE_HOST,
)

client = LangfuseClient(
    secret_key=LANGFUSE_SECRET,
    public_key=LANGFUSE_PUBLIC,
    host=LANGFUSE_HOST,
)

# -------- Middleware for HTTP calls --------

class LangfuseMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        trace = langfuse.trace(
            name=f"{request.method} {request.url.path}",
            input={"query_params": dict(request.query_params)},
        )

        response = await call_next(request)

        trace.end(
            output={"status_code": response.status_code},
        )

        return response


def setup_langfuse_middleware(app):
    """Attach Langfuse middleware to FastAPI."""
    app.add_middleware(LangfuseMiddleware)


from .logging_config import configure_logging, logger
from .langfuse_config import setup_langfuse_middleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()   # logging setup
    setup_langfuse_middleware(app)  # Langfuse tracing
    create_db_and_tables()
    yield

from langfuse_config import langfuse

async def run_agent(query: str):
    trace = langfuse.trace(name="agent.run", input={"query": query})

    step1 = trace.span("RAG")
    context = await asyncio.to_thread(query_documents, query)
    step1.end()
    
    step2 = trace.span("Web Search")
    web_results = await asyncio.to_thread(web_search, query)
    step2.end()

    # ... other steps ...

    trace.end(output={"answer": final_answer})

    return {"answer": final_answer}

