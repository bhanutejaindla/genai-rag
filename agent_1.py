<div class="progress-container">
  <div class="progress-card">
    <div class="header">
      <h2>Job Progress</h2>
      <a routerLink="/dashboard" class="btn-link">← Back to Dashboard</a>
    </div>

    <div *ngIf="loading" class="loading">Loading job status...</div>
    <div *ngIf="error" class="error-message">{{ error }}</div>

    <div *ngIf="job && !loading" class="job-details">
      <!-- Overall Progress -->
      <div class="progress-section">
        <div class="progress-header">
          <h3>Overall Progress</h3>
          <span class="job-status" [ngClass]="'status-' + job.status">
            {{ job.status }}
          </span>
        </div>
        <div class="progress-bar-container">
          <div class="progress-bar">
            <div class="progress-fill" [style.width.%]="job.progress"></div>
          </div>
          <span class="progress-text">{{ job.progress }}%</span>
        </div>
      </div>

      <!-- Job Info -->
      <div class="info-section">
        <div class="info-item">
          <span class="info-label">Job ID:</span>
          <span class="info-value">#{{ job.id }}</span>
        </div>
        <div class="info-item">
          <span class="info-label">Type:</span>
          <span class="info-value">{{ job.type }}</span>
        </div>
        <div class="info-item">
          <span class="info-label">Created:</span>
          <span class="info-value">{{ job.created_at | date:'medium' }}</span>
        </div>
        <div class="info-item" *ngIf="job.started_at">
          <span class="info-label">Started:</span>
          <span class="info-value">{{ job.started_at | date:'medium' }}</span>
        </div>
      </div>

      <!-- Tool Status -->
      <div class="tools-section" *ngIf="job.tasks && job.tasks.length > 0">
        <h3>Tool Status</h3>
        <div class="tools-list">
          <div *ngFor="let task of job.tasks" class="tool-item">
            <div class="tool-header">
              <span class="tool-name">{{ task.name || 'Unknown Tool' }}</span>
              <span class="tool-status" [ngClass]="'status-' + getTaskStatus(task)">
                {{ getTaskStatus(task) }}
              </span>
            </div>
            <div class="tool-progress" *ngIf="getTaskStatus(task) === 'running'">
              <div class="progress-bar small">
                <div class="progress-fill" [style.width.%]="getTaskProgress(task)"></div>
              </div>
              <span class="progress-text">{{ getTaskProgress(task) }}%</span>
            </div>
            <div class="tool-message" *ngIf="task.message">
              {{ task.message }}
            </div>
          </div>
        </div>
      </div>

      <!-- Actions -->
      <div class="actions-section">
        <button
          *ngIf="job.status === 'running' || job.status === 'pending'"
          class="btn btn-danger"
          (click)="cancelJob()"
        >
          Cancel Job
        </button>
        <button
          *ngIf="job.status === 'completed'"
          class="btn btn-primary"
          [routerLink]="['/reports', job.id]"
        >
          View Report
        </button>
      </div>
    </div>
  </div>
</div>


.progress-container {
  min-height: 100vh;
  background: #f5f7fa;
  padding: 40px 20px;
}

.progress-card {
  max-width: 900px;
  margin: 0 auto;
  background: white;
  border-radius: 12px;
  padding: 40px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 30px;
}

.header h2 {
  margin: 0;
  color: #333;
}

.btn-link {
  color: #667eea;
  text-decoration: none;
  font-size: 14px;
}

.btn-link:hover {
  text-decoration: underline;
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

.job-details {
  display: flex;
  flex-direction: column;
  gap: 30px;
}

.progress-section {
  padding: 20px;
  background: #f8f9fa;
  border-radius: 8px;
}

.progress-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
}

.progress-header h3 {
  margin: 0;
  color: #333;
}

.job-status {
  padding: 6px 12px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 500;
  text-transform: capitalize;
}

.status-pending {
  background: #fff3cd;
  color: #856404;
}

.status-running {
  background: #d1ecf1;
  color: #0c5460;
}

.status-completed {
  background: #d4edda;
  color: #155724;
}

.status-failed {
  background: #f8d7da;
  color: #721c24;
}

.progress-bar-container {
  display: flex;
  align-items: center;
  gap: 15px;
}

.progress-bar {
  flex: 1;
  height: 20px;
  background: #e9ecef;
  border-radius: 10px;
  overflow: hidden;
  position: relative;
}

.progress-bar.small {
  height: 12px;
  width: 200px;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
  transition: width 0.5s ease;
  border-radius: 10px;
}

.progress-text {
  font-size: 14px;
  font-weight: 500;
  color: #333;
  min-width: 50px;
  text-align: right;
}

.info-section {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
  padding: 20px;
  background: #f8f9fa;
  border-radius: 8px;
}

.info-item {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.info-label {
  font-size: 12px;
  color: #666;
  text-transform: uppercase;
  font-weight: 500;
}

.info-value {
  font-size: 16px;
  color: #333;
  font-weight: 500;
}

.tools-section h3 {
  margin: 0 0 20px 0;
  color: #333;
}

.tools-list {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.tool-item {
  padding: 20px;
  background: #f8f9fa;
  border-radius: 8px;
  border-left: 4px solid #667eea;
}

.tool-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.tool-name {
  font-weight: 500;
  color: #333;
  text-transform: capitalize;
}

.tool-status {
  padding: 4px 10px;
  border-radius: 12px;
  font-size: 11px;
  font-weight: 500;
  text-transform: capitalize;
}

.tool-progress {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 10px;
}

.tool-message {
  margin-top: 10px;
  font-size: 13px;
  color: #666;
  font-style: italic;
}

.actions-section {
  display: flex;
  justify-content: flex-end;
  gap: 15px;
  padding-top: 20px;
  border-top: 1px solid #eee;
}

.btn {
  padding: 12px 24px;
  border: none;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s;
}

.btn-primary {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.btn-primary:hover {
  opacity: 0.9;
}

.btn-danger {
  background: #e74c3c;
  color: white;
}

.btn-danger:hover {
  background: #c0392b;
}


import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, Router, RouterModule } from '@angular/router';
import { ApiService, Job } from '../../../services/api.service';
import { interval, Subscription } from 'rxjs';
import { switchMap } from 'rxjs/operators';

@Component({
  selector: 'app-progress',
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './progress.component.html',
  styleUrls: ['./progress.component.css']
})
export class ProgressComponent implements OnInit, OnDestroy {
  jobId!: number;
  job: Job | null = null;
  loading: boolean = true;
  error: string = '';
  private subscription?: Subscription;
  private pollInterval = 2000; // Poll every 2 seconds

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private apiService: ApiService
  ) {}

  ngOnInit() {
    this.jobId = +this.route.snapshot.paramMap.get('id')!;
    this.loadJob();
    this.startPolling();
  }

  ngOnDestroy() {
    if (this.subscription) {
      this.subscription.unsubscribe();
    }
  }

  loadJob() {
    this.apiService.getJob(this.jobId).subscribe({
      next: (job) => {
        this.job = job;
        this.loading = false;

        // Redirect to report view if completed
        if (job.status === 'completed') {
          setTimeout(() => {
            this.router.navigate(['/reports', job.id]);
          }, 2000);
        }
      },
      error: (err) => {
        this.error = 'Failed to load job progress';
        this.loading = false;
      }
    });
  }

  startPolling() {
    this.subscription = interval(this.pollInterval)
      .pipe(
        switchMap(() => this.apiService.getJob(this.jobId))
      )
      .subscribe({
        next: (job) => {
          this.job = job;
          if (job.status === 'completed' || job.status === 'failed') {
            if (this.subscription) {
              this.subscription.unsubscribe();
            }
          }
        },
        error: (err) => {
          console.error('Error polling job:', err);
        }
      });
  }

  cancelJob() {
    if (confirm('Are you sure you want to cancel this job?')) {
      this.apiService.cancelJob(this.jobId).subscribe({
        next: () => {
          this.router.navigate(['/dashboard']);
        },
        error: (err) => {
          this.error = 'Failed to cancel job';
        }
      });
    }
  }

  getTaskStatus(task: any): string {
    if (!task) return 'pending';
    return task.status || 'pending';
  }

  getTaskProgress(task: any): number {
    if (!task) return 0;
    return task.progress || 0;
  }
}

//report-edit
<div class="edit-container">
  <div class="edit-header">
    <div class="header-content">
      <div>
        <h1>Edit Report #{{ reportId }}</h1>
        <a [routerLink]="['/reports', reportId]" class="btn-link">← Back to Report</a>
      </div>
      <div class="header-actions">
        <button class="btn btn-secondary" (click)="downloadReport('docx')">Download DOCX</button>
        <button class="btn btn-secondary" (click)="downloadReport('pdf')">Download PDF</button>
        <button class="btn btn-secondary" (click)="cancel()">Cancel</button>
        <button class="btn btn-primary" (click)="saveReport()" [disabled]="saving">
          {{ saving ? 'Saving...' : 'Save Changes' }}
        </button>
      </div>
    </div>
  </div>

  <div class="edit-content">
    <div *ngIf="loading" class="loading">Loading report...</div>
    <div *ngIf="error" class="error-message">{{ error }}</div>

    <div *ngIf="report && !loading" class="editor-wrapper">
      <div class="editor-toolbar">
        <span class="toolbar-info">Editing report content</span>
        <span class="toolbar-hint">Changes will be saved as a new version</span>
      </div>

      <textarea
        [(ngModel)]="editedContent"
        class="content-editor"
        placeholder="Enter report content..."
      ></textarea>

      <div class="editor-footer">
        <div class="char-count">
          {{ editedContent.length }} characters
        </div>
      </div>
    </div>
  </div>
</div>

.edit-container {
  min-height: 100vh;
  background: #f5f7fa;
}

.edit-header {
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
  flex-wrap: wrap;
  gap: 20px;
}

.header-content h1 {
  margin: 0 0 10px 0;
  color: #333;
}

.header-actions {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.edit-content {
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

.editor-wrapper {
  background: white;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  overflow: hidden;
}

.editor-toolbar {
  padding: 15px 20px;
  background: #f8f9fa;
  border-bottom: 1px solid #eee;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.toolbar-info {
  font-weight: 500;
  color: #333;
}

.toolbar-hint {
  font-size: 12px;
  color: #666;
}

.content-editor {
  width: 100%;
  min-height: 600px;
  padding: 30px;
  border: none;
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  font-size: 14px;
  line-height: 1.8;
  resize: vertical;
  box-sizing: border-box;
}

.content-editor:focus {
  outline: none;
}

.editor-footer {
  padding: 15px 20px;
  background: #f8f9fa;
  border-top: 1px solid #eee;
  display: flex;
  justify-content: flex-end;
}

.char-count {
  font-size: 12px;
  color: #666;
}

.btn {
  padding: 12px 24px;
  border: none;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s;
}

.btn-primary {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.btn-primary:hover:not(:disabled) {
  opacity: 0.9;
}

.btn-primary:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn-secondary {
  background: #6c757d;
  color: white;
}

.btn-secondary:hover {
  background: #5a6268;
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
import { ActivatedRoute, Router, RouterModule } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { ApiService, Report } from '../../../services/api.service';

@Component({
  selector: 'app-report-edit',
  standalone: true,
  imports: [CommonModule, RouterModule, FormsModule],
  templateUrl: './report-edit.component.html',
  styleUrls: ['./report-edit.component.css']
})
export class ReportEditComponent implements OnInit {
  reportId!: number;
  report: Report | null = null;
  editedContent: string = '';
  loading: boolean = true;
  saving: boolean = false;
  error: string = '';

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private apiService: ApiService
  ) {}

  ngOnInit() {
    this.reportId = +this.route.snapshot.paramMap.get('id')!;
    this.loadReport();
  }

  loadReport() {
    this.apiService.getReport(this.reportId).subscribe({
      next: (report) => {
        this.report = report;
        this.editedContent = report.content;
        this.loading = false;
      },
      error: (err) => {
        this.error = 'Failed to load report';
        this.loading = false;
      }
    });
  }

  saveReport() {
    if (!this.editedContent.trim()) {
      this.error = 'Report content cannot be empty';
      return;
    }

    this.saving = true;
    this.error = '';

    this.apiService.updateReport(this.reportId, this.editedContent).subscribe({
      next: (updatedReport) => {
        this.saving = false;
        this.router.navigate(['/reports', this.reportId]);
      },
      error: (err) => {
        this.error = err.error?.detail || 'Failed to save report';
        this.saving = false;
      }
    });
  }

  downloadReport(format: 'pdf' | 'docx') {
    this.apiService.downloadReport(this.reportId, format).subscribe({
      next: (blob) => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `report_${this.reportId}.${format}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      },
      error: (err) => {
        alert('Failed to download report');
      }
    });
  }

  cancel() {
    if (confirm('Are you sure you want to discard your changes?')) {
      this.router.navigate(['/reports', this.reportId]);
    }
  }
}


//report-view

<div class="report-container">
  <div class="report-header">
    <div class="header-content">
      <div>
        <h1>Report #{{ reportId }}</h1>
        <a routerLink="/dashboard" class="btn-link">← Back to Dashboard</a>
      </div>
      <div class="header-actions">
        <button class="btn btn-secondary" (click)="editReport()">Edit</button>
        <button class="btn btn-secondary" (click)="downloadReport('docx')">Download DOCX</button>
        <button class="btn btn-secondary" (click)="downloadReport('pdf')">Download PDF</button>
        <button class="btn btn-primary" (click)="toggleChat()">
          {{ showChat ? 'Hide' : 'Show' }} Chat
        </button>
      </div>
    </div>
  </div>

  <div class="report-content-wrapper">
    <div class="report-content" [class.with-chat]="showChat">
      <div *ngIf="loading" class="loading">Loading report...</div>
      <div *ngIf="error" class="error-message">{{ error }}</div>

      <div *ngIf="report && !loading" class="report-body">
        <div class="report-meta">
          <span class="meta-item">Created: {{ report.created_at | date:'medium' }}</span>
          <span class="meta-item" *ngIf="report.citations">
            Citations: {{ report.citations.length }}
          </span>
        </div>

        <div class="report-text" [innerHTML]="parseContent(report.content)"></div>

        <!-- Citations Section -->
        <div *ngIf="report.citations && report.citations.length > 0" class="citations-section">
          <h3>Citations</h3>
          <div class="citations-list">
            <div *ngFor="let citation of report.citations; let i = index" class="citation-item">
              <span class="citation-number">[{{ i + 1 }}]</span>
              <div class="citation-content">
                <div class="citation-title" *ngIf="citation.title">{{ citation.title }}</div>
                <div class="citation-url" *ngIf="citation.url">
                  <a [href]="citation.url" target="_blank">{{ citation.url }}</a>
                </div>
                <div class="citation-snippet" *ngIf="citation.snippet">{{ citation.snippet }}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Chat Panel -->
    <div class="chat-panel" *ngIf="showChat">
      <div class="chat-header">
        <h3>Chat about this Report</h3>
        <p class="chat-subtitle">Ask questions about the report content</p>
      </div>

      <div class="chat-messages">
        <div *ngFor="let message of messages" class="message" [ngClass]="'message-' + message.role">
          <div class="message-header">
            <span class="message-role">{{ message.role === 'user' ? 'You' : 'Assistant' }}</span>
            <span class="message-time">{{ message.timestamp | date:'short' }}</span>
          </div>
          <div class="message-content">{{ message.content }}</div>
        </div>
        <div *ngIf="chatLoading" class="message message-assistant">
          <div class="message-content">Thinking...</div>
        </div>
      </div>

      <div class="chat-input-area">
        <form (ngSubmit)="sendChatMessage()" class="chat-form">
          <input
            type="text"
            [(ngModel)]="chatInput"
            name="chatInput"
            class="chat-input"
            placeholder="Ask a question about the report..."
            [disabled]="chatLoading"
          />
          <button type="submit" class="btn btn-primary" [disabled]="chatLoading || !chatInput.trim()">
            Send
          </button>
        </form>
      </div>
    </div>
  </div>
</div>


.report-container {
  min-height: 100vh;
  background: #f5f7fa;
}

.report-header {
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
  flex-wrap: wrap;
  gap: 20px;
}

.header-content h1 {
  margin: 0 0 10px 0;
  color: #333;
}

.header-actions {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.report-content-wrapper {
  max-width: 1400px;
  margin: 0 auto;
  padding: 0 20px;
  display: flex;
  gap: 30px;
}

.report-content {
  flex: 1;
  background: white;
  border-radius: 12px;
  padding: 40px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.report-content.with-chat {
  max-width: 60%;
}

.loading, .error-message {
  text-align: center;
  padding: 40px;
  color: #666;
}

.error-message {
  color: #e74c3c;
  background: #fee;
  border-radius: 6px;
}

.report-meta {
  display: flex;
  gap: 20px;
  padding-bottom: 20px;
  border-bottom: 1px solid #eee;
  margin-bottom: 30px;
  font-size: 14px;
  color: #666;
}

.meta-item {
  padding: 6px 12px;
  background: #f8f9fa;
  border-radius: 4px;
}

.report-text {
  line-height: 1.8;
  color: #333;
  font-size: 16px;
  margin-bottom: 40px;
}

.report-text .citation {
  background: #fff3cd;
  padding: 2px 6px;
  border-radius: 3px;
  font-size: 12px;
  font-weight: 500;
  color: #856404;
  cursor: pointer;
}

.report-text .citation:hover {
  background: #ffeaa7;
}

.citations-section {
  margin-top: 40px;
  padding-top: 30px;
  border-top: 2px solid #eee;
}

.citations-section h3 {
  margin: 0 0 20px 0;
  color: #333;
}

.citations-list {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.citation-item {
  display: flex;
  gap: 15px;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 6px;
  border-left: 4px solid #667eea;
}

.citation-number {
  font-weight: 600;
  color: #667eea;
  min-width: 40px;
}

.citation-content {
  flex: 1;
}

.citation-title {
  font-weight: 500;
  color: #333;
  margin-bottom: 5px;
}

.citation-url {
  margin-bottom: 5px;
}

.citation-url a {
  color: #667eea;
  text-decoration: none;
  font-size: 14px;
}

.citation-url a:hover {
  text-decoration: underline;
}

.citation-snippet {
  color: #666;
  font-size: 14px;
  font-style: italic;
  margin-top: 5px;
}

.chat-panel {
  width: 400px;
  background: white;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
  height: calc(100vh - 200px);
  position: sticky;
  top: 20px;
}

.chat-header {
  padding: 20px;
  border-bottom: 1px solid #eee;
}

.chat-header h3 {
  margin: 0 0 5px 0;
  color: #333;
}

.chat-subtitle {
  margin: 0;
  font-size: 12px;
  color: #666;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.message {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.message-user {
  align-items: flex-end;
}

.message-assistant {
  align-items: flex-start;
}

.message-header {
  display: flex;
  gap: 10px;
  font-size: 11px;
  color: #999;
}

.message-role {
  font-weight: 500;
}

.message-content {
  padding: 12px 16px;
  border-radius: 12px;
  max-width: 80%;
  font-size: 14px;
  line-height: 1.5;
}

.message-user .message-content {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.message-assistant .message-content {
  background: #f0f0f0;
  color: #333;
}

.chat-input-area {
  padding: 20px;
  border-top: 1px solid #eee;
}

.chat-form {
  display: flex;
  gap: 10px;
}

.chat-input {
  flex: 1;
  padding: 12px;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 14px;
}

.chat-input:focus {
  outline: none;
  border-color: #667eea;
}

.btn {
  padding: 12px 24px;
  border: none;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s;
}

.btn-primary {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.btn-primary:hover:not(:disabled) {
  opacity: 0.9;
}

.btn-primary:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn-secondary {
  background: #6c757d;
  color: white;
}

.btn-secondary:hover {
  background: #5a6268;
}

.btn-link {
  color: #667eea;
  text-decoration: none;
  font-size: 14px;
}

.btn-link:hover {
  text-decoration: underline;
}

@media (max-width: 1200px) {
  .report-content-wrapper {
    flex-direction: column;
  }

  .report-content.with-chat {
    max-width: 100%;
  }

  .chat-panel {
    width: 100%;
    height: 500px;
    position: relative;
  }
}

import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, Router, RouterModule } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { ApiService, Report } from '../../../services/api.service';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

@Component({
  selector: 'app-report-view',
  standalone: true,
  imports: [CommonModule, RouterModule, FormsModule],
  templateUrl: './report-view.component.html',
  styleUrls: ['./report-view.component.css']
})
export class ReportViewComponent implements OnInit {
  reportId!: number;
  report: Report | null = null;
  loading: boolean = true;
  error: string = '';

  // Chat
  messages: Message[] = [];
  chatInput: string = '';
  chatLoading: boolean = false;
  showChat: boolean = false;

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private apiService: ApiService
  ) {}

  ngOnInit() {
    this.reportId = +this.route.snapshot.paramMap.get('id')!;
    this.loadReport();
  }

  loadReport() {
    this.apiService.getReport(this.reportId).subscribe({
      next: (report) => {
        this.report = report;
        this.loading = false;
      },
      error: (err) => {
        this.error = 'Failed to load report';
        this.loading = false;
      }
    });
  }

  toggleChat() {
    this.showChat = !this.showChat;
  }

  sendChatMessage() {
    if (!this.chatInput.trim()) return;

    const userMessage: Message = {
      role: 'user',
      content: this.chatInput,
      timestamp: new Date()
    };
    this.messages.push(userMessage);
    this.chatInput = '';
    this.chatLoading = true;

    this.apiService.sendChatMessage(userMessage.content, this.reportId).subscribe({
      next: (response) => {
        const assistantMessage: Message = {
          role: 'assistant',
          content: response.response,
          timestamp: new Date()
        };
        this.messages.push(assistantMessage);
        this.chatLoading = false;
      },
      error: (err) => {
        const errorMessage: Message = {
          role: 'assistant',
          content: 'Sorry, I encountered an error. Please try again.',
          timestamp: new Date()
        };
        this.messages.push(errorMessage);
        this.chatLoading = false;
      }
    });
  }

  downloadReport(format: 'pdf' | 'docx') {
    this.apiService.downloadReport(this.reportId, format).subscribe({
      next: (blob) => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `report_${this.reportId}.${format}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      },
      error: (err) => {
        alert('Failed to download report');
      }
    });
  }

  editReport() {
    this.router.navigate(['/reports', this.reportId, 'edit']);
  }

  parseContent(content: string): string {
    // Simple markdown-like parsing for citations
    return content
      .replace(/\[citation:(\d+)\]/g, '<span class="citation">[Citation $1]</span>')
      .replace(/\n/g, '<br>');
  }
}

