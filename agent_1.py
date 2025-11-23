<div class="create-job-container">
  <div class="create-job-card">
    <div class="header">
      <h2>Create Research Job</h2>
      <a routerLink="/dashboard" class="btn-link">← Back to Dashboard</a>
    </div>

    <form (ngSubmit)="onSubmit()" #jobForm="ngForm">
      <!-- Topic Input -->
      <div class="form-group">
        <label for="topic">Research Topic *</label>
        <textarea
          id="topic"
          name="topic"
          [(ngModel)]="topic"
          required
          minlength="3"
          class="form-control"
          rows="4"
          placeholder="Enter the research topic or question you want to investigate..."
        ></textarea>
        <div *ngIf="errors['topic']" class="error-message">
          {{ errors['topic'] }}
        </div>
      </div>

      <!-- Document Upload -->
      <div class="form-group">
        <label for="documents">Upload Documents (Optional)</label>
        <div class="file-upload-area">
          <input
            type="file"
            id="documents"
            name="documents"
            (change)="onFileSelected($event)"
            multiple
            accept=".pdf,.docx,.txt"
            class="file-input"
          />
          <label for="documents" class="file-label">
            <span class="upload-icon">📄</span>
            <span>Click to upload or drag and drop</span>
            <span class="file-hint">PDF, DOCX, TXT files only</span>
          </label>
        </div>
        <div *ngIf="errors['documents']" class="error-message">
          {{ errors['documents'] }}
        </div>

        <!-- Uploaded Files List -->
        <div *ngIf="documents.length > 0" class="files-list">
          <div *ngFor="let file of documents; let i = index" class="file-item">
            <span class="file-name">{{ file.name }}</span>
            <span class="file-size">{{ formatFileSize(file.size) }}</span>
            <button
              type="button"
              class="btn-remove"
              (click)="removeDocument(i)"
            >
              ✕
            </button>
          </div>
        </div>
      </div>

      <!-- Tool Configuration -->
      <div class="form-group">
        <label>Tool Configuration *</label>
        <div class="tools-config">
          <label class="tool-item">
            <input
              type="checkbox"
              [(ngModel)]="toolConfig.web_search"
              name="web_search"
            />
            <span class="tool-label">
              <strong>Web Search</strong>
              <span class="tool-desc">Search the web for relevant information</span>
            </span>
          </label>

          <label class="tool-item">
            <input
              type="checkbox"
              [(ngModel)]="toolConfig.rag"
              name="rag"
            />
            <span class="tool-label">
              <strong>RAG (Retrieval Augmented Generation)</strong>
              <span class="tool-desc">Query uploaded documents and knowledge base</span>
            </span>
          </label>

          <label class="tool-item">
            <input
              type="checkbox"
              [(ngModel)]="toolConfig.compliance"
              name="compliance"
            />
            <span class="tool-label">
              <strong>Compliance Check</strong>
              <span class="tool-desc">Redact PII and ensure compliance</span>
            </span>
          </label>

          <label class="tool-item">
            <input
              type="checkbox"
              [(ngModel)]="toolConfig.citation_validation"
              name="citation_validation"
            />
            <span class="tool-label">
              <strong>Citation Validation</strong>
              <span class="tool-desc">Verify and validate citations in the report</span>
            </span>
          </label>
        </div>
        <div *ngIf="errors['tools']" class="error-message">
          {{ errors['tools'] }}
        </div>
      </div>

      <!-- Submit Error -->
      <div *ngIf="errors['submit']" class="error-message">
        {{ errors['submit'] }}
      </div>

      <!-- Submit Button -->
      <div class="form-actions">
        <button
          type="button"
          class="btn btn-secondary"
          routerLink="/dashboard"
        >
          Cancel
        </button>
        <button
          type="submit"
          class="btn btn-primary"
          [disabled]="loading || !jobForm.valid"
        >
          {{ loading ? 'Creating Job...' : 'Create Research Job' }}
        </button>
      </div>
    </form>
  </div>
</div>


.create-job-container {
  min-height: 100vh;
  background: #f5f7fa;
  padding: 40px 20px;
}

.create-job-card {
  max-width: 800px;
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

.form-group {
  margin-bottom: 30px;
}

.form-group label {
  display: block;
  margin-bottom: 8px;
  color: #555;
  font-weight: 500;
  font-size: 14px;
}

.form-control {
  width: 100%;
  padding: 12px;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 14px;
  font-family: inherit;
  transition: border-color 0.3s;
  box-sizing: border-box;
}

.form-control:focus {
  outline: none;
  border-color: #667eea;
}

.file-upload-area {
  position: relative;
  border: 2px dashed #ddd;
  border-radius: 8px;
  padding: 40px;
  text-align: center;
  transition: all 0.3s;
  cursor: pointer;
}

.file-upload-area:hover {
  border-color: #667eea;
  background: #f8f9ff;
}

.file-input {
  position: absolute;
  width: 100%;
  height: 100%;
  top: 0;
  left: 0;
  opacity: 0;
  cursor: pointer;
}

.file-label {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  pointer-events: none;
}

.upload-icon {
  font-size: 48px;
}

.file-hint {
  font-size: 12px;
  color: #999;
}

.files-list {
  margin-top: 15px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.file-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px;
  background: #f8f9fa;
  border-radius: 6px;
  font-size: 14px;
}

.file-name {
  flex: 1;
  color: #333;
}

.file-size {
  color: #666;
  margin: 0 15px;
}

.btn-remove {
  background: #e74c3c;
  color: white;
  border: none;
  border-radius: 4px;
  width: 24px;
  height: 24px;
  cursor: pointer;
  font-size: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.btn-remove:hover {
  background: #c0392b;
}

.tools-config {
  display: flex;
  flex-direction: column;
  gap: 15px;
  padding: 20px;
  background: #f8f9fa;
  border-radius: 8px;
}

.tool-item {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  cursor: pointer;
  padding: 12px;
  border-radius: 6px;
  transition: background 0.3s;
}

.tool-item:hover {
  background: white;
}

.tool-item input[type="checkbox"] {
  margin-top: 4px;
  cursor: pointer;
}

.tool-label {
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex: 1;
}

.tool-label strong {
  color: #333;
  font-size: 14px;
}

.tool-desc {
  color: #666;
  font-size: 12px;
}

.error-message {
  color: #e74c3c;
  margin-top: 8px;
  font-size: 14px;
  padding: 8px;
  background: #fee;
  border-radius: 4px;
}

.form-actions {
  display: flex;
  justify-content: flex-end;
  gap: 15px;
  margin-top: 30px;
  padding-top: 30px;
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


import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router, RouterModule } from '@angular/router';
import { ApiService, CreateJobRequest } from '../../../services/api.service';

@Component({
  selector: 'app-create-job',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterModule],
  templateUrl: './create-job.component.html',
  styleUrls: ['./create-job.component.css']
})
export class CreateJobComponent {
  topic: string = '';
  documents: File[] = [];
  toolConfig: { [key: string]: boolean } = {
    web_search: true,
    rag: true,
    compliance: true,
    citation_validation: true
  };

  errors: { [key: string]: string } = {};
  loading: boolean = false;

  constructor(
    private apiService: ApiService,
    private router: Router
  ) {}

  onFileSelected(event: any) {
    const files = Array.from(event.target.files) as File[];
    this.documents = [...this.documents, ...files];
    this.errors['documents'] = '';
  }

  removeDocument(index: number) {
    this.documents.splice(index, 1);
  }

  validate(): boolean {
    this.errors = {};

    if (!this.topic || this.topic.trim().length < 3) {
      this.errors['topic'] = 'Topic must be at least 3 characters';
      return false;
    }

    if (this.documents.length > 0) {
      const invalidFiles = this.documents.filter(
        file => !file.name.match(/\.(pdf|docx|txt)$/i)
      );
      if (invalidFiles.length > 0) {
        this.errors['documents'] = 'Only PDF, DOCX, and TXT files are allowed';
        return false;
      }
    }

    const hasToolEnabled = Object.values(this.toolConfig).some(enabled => enabled);
    if (!hasToolEnabled) {
      this.errors['tools'] = 'At least one tool must be enabled';
      return false;
    }

    return true;
  }

  onSubmit() {
    if (!this.validate()) {
      return;
    }

    this.loading = true;

    const jobData: CreateJobRequest = {
      topic: this.topic.trim(),
      documents: this.documents.length > 0 ? this.documents : undefined,
      tool_config: this.toolConfig
    };

    this.apiService.createJob(jobData).subscribe({
      next: (response) => {
        this.router.navigate(['/jobs', response.job_id, 'progress']);
      },
      error: (err) => {
        this.errors['submit'] = err.error?.detail || 'Failed to create job. Please try again.';
        this.loading = false;
      }
    });
  }

  formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  }
}


import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';
import { AuthService } from './auth.service';

export interface Job {
  id: number;
  type: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  user_id: number;
  progress: number;
  tasks?: any[];
  created_at: string;
  started_at?: string;
  updated_at: string;
}

export interface Report {
  id: number;
  job_id: number;
  content: string;
  citations?: any[];
  created_at: string;
}

export interface CreateJobRequest {
  topic: string;
  documents?: File[];
  tool_config?: {
    [key: string]: boolean;
  };
}

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  constructor(
    private http: HttpClient,
    private authService: AuthService
  ) {}

  private getHeaders(): HttpHeaders {
    const authHeaders = this.authService.getAuthHeaders();
    return new HttpHeaders({
      ...authHeaders,
      'Content-Type': 'application/json'
    });
  }

  private getFormHeaders(): HttpHeaders {
    const authHeaders = this.authService.getAuthHeaders();
    return new HttpHeaders(authHeaders);
  }

  // Jobs
  createJob(jobData: CreateJobRequest): Observable<{ job_id: number; status: string }> {
    const formData = new FormData();
    formData.append('topic', jobData.topic);
    if (jobData.documents) {
      jobData.documents.forEach(file => {
        formData.append('documents', file);
      });
    }
    if (jobData.tool_config) {
      formData.append('tool_config', JSON.stringify(jobData.tool_config));
    }

    return this.http.post<{ job_id: number; status: string }>(
      `${environment.apiBaseUrl}/jobs`,
      formData,
      { headers: this.getFormHeaders() }
    );
  }

  getJob(jobId: number): Observable<Job> {
    return this.http.get<Job>(
      `${environment.apiBaseUrl}/jobs/${jobId}`,
      { headers: this.getHeaders() }
    );
  }

  getJobs(params?: { page?: number; limit?: number; status?: string }): Observable<{ jobs: Job[]; total: number }> {
    let httpParams = new HttpParams();
    if (params?.page) httpParams = httpParams.set('page', params.page);
    if (params?.limit) httpParams = httpParams.set('limit', params.limit);
    if (params?.status) httpParams = httpParams.set('status', params.status);

    return this.http.get<{ jobs: Job[]; total: number }>(
      `${environment.apiBaseUrl}/jobs`,
      { headers: this.getHeaders(), params: httpParams }
    );
  }

  cancelJob(jobId: number): Observable<void> {
    return this.http.post<void>(
      `${environment.apiBaseUrl}/jobs/${jobId}/cancel`,
      {},
      { headers: this.getHeaders() }
    );
  }

  // Reports
  getReport(reportId: number): Observable<Report> {
    return this.http.get<Report>(
      `${environment.apiBaseUrl}/reports/${reportId}`,
      { headers: this.getHeaders() }
    );
  }

  getReports(jobId?: number): Observable<Report[]> {
    let url = `${environment.apiBaseUrl}/reports`;
    if (jobId) {
      url += `?job_id=${jobId}`;
    }
    return this.http.get<Report[]>(url, { headers: this.getHeaders() });
  }

  downloadReport(reportId: number, format: 'pdf' | 'docx'): Observable<Blob> {
    return this.http.get(
      `${environment.apiBaseUrl}/reports/${reportId}/download?format=${format}`,
      { headers: this.getHeaders(), responseType: 'blob' }
    );
  }

  updateReport(reportId: number, content: string): Observable<Report> {
    return this.http.put<Report>(
      `${environment.apiBaseUrl}/reports/${reportId}`,
      { content },
      { headers: this.getHeaders() }
    );
  }

  // Chat
  sendChatMessage(message: string, reportId?: number): Observable<{ response: string }> {
    const body: any = { message };
    if (reportId) {
      body.report_id = reportId;
    }
    return this.http.post<{ response: string }>(
      `${environment.apiBaseUrl}/chat`,
      body,
      { headers: this.getHeaders() }
    );
  }

  // Admin
  getAdminMetrics(): Observable<any> {
    return this.http.get(
      `${environment.apiBaseUrl}/admin/metrics`,
      { headers: this.getHeaders() }
    );
  }

  getToolRegistry(): Observable<any[]> {
    return this.http.get<any[]>(
      `${environment.apiBaseUrl}/admin/tools`,
      { headers: this.getHeaders() }
    );
  }

  updateToolQuota(toolId: string, quota: number): Observable<void> {
    return this.http.put<void>(
      `${environment.apiBaseUrl}/admin/tools/${toolId}/quota`,
      { quota },
      { headers: this.getHeaders() }
    );
  }

  // Document upload
  uploadDocument(file: File): Observable<any> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post(
      `${environment.apiBaseUrl}/ingest`,
      formData,
      { headers: this.getFormHeaders() }
    );
  }
}

