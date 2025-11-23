<div class="auth-container">
  <div class="auth-card">
    <h2>Login</h2>
    <form (ngSubmit)="onSubmit()" #loginForm="ngForm">
      <div class="form-group">
        <label for="email">Email</label>
        <input
          type="email"
          id="email"
          name="email"
          [(ngModel)]="email"
          required
          class="form-control"
          placeholder="Enter your email"
        />
      </div>

      <div class="form-group">
        <label for="password">Password</label>
        <input
          type="password"
          id="password"
          name="password"
          [(ngModel)]="password"
          required
          class="form-control"
          placeholder="Enter your password"
        />
      </div>

      <div *ngIf="error" class="error-message">
        {{ error }}
      </div>

      <button
        type="submit"
        class="btn btn-primary"
        [disabled]="loading || !loginForm.valid"
      >
        {{ loading ? 'Logging in...' : 'Login' }}
      </button>
    </form>

    <p class="auth-link">
      Don't have an account? <a routerLink="/register">Sign up</a>
    </p>
  </div>
</div>





.auth-container {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 20px;
}

.auth-card {
  background: white;
  border-radius: 12px;
  padding: 40px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
  width: 100%;
  max-width: 400px;
}

.auth-card h2 {
  margin: 0 0 30px 0;
  color: #333;
  text-align: center;
}

.form-group {
  margin-bottom: 20px;
}

.form-group label {
  display: block;
  margin-bottom: 8px;
  color: #555;
  font-weight: 500;
}

.form-control {
  width: 100%;
  padding: 12px;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 14px;
  transition: border-color 0.3s;
  box-sizing: border-box;
}

.form-control:focus {
  outline: none;
  border-color: #667eea;
}

.btn {
  width: 100%;
  padding: 12px;
  border: none;
  border-radius: 6px;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
  transition: background-color 0.3s;
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

.error-message {
  color: #e74c3c;
  margin-bottom: 15px;
  padding: 10px;
  background: #fee;
  border-radius: 6px;
  font-size: 14px;
}

.auth-link {
  text-align: center;
  margin-top: 20px;
  color: #666;
}

.auth-link a {
  color: #667eea;
  text-decoration: none;
  font-weight: 500;
}

.auth-link a:hover {
  text-decoration: underline;
}


import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router, RouterModule } from '@angular/router';
import { AuthService } from '../../../services/auth.service';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterModule],
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent {
  email: string = '';
  password: string = '';
  error: string = '';
  loading: boolean = false;

  constructor(
    private authService: AuthService,
    private router: Router
  ) {}

  onSubmit() {
    if (!this.email || !this.password) {
      this.error = 'Please fill in all fields';
      return;
    }

    this.loading = true;
    this.error = '';

    this.authService.login({
      username: this.email,
      password: this.password
    }).subscribe({
      next: () => {
        this.router.navigate(['/dashboard']);
      },
      error: (err) => {
        this.error = err.error?.detail || 'Login failed. Please check your credentials.';
        this.loading = false;
      }
    });
  }
}



<div class="auth-container">
  <div class="auth-card">
    <h2>Sign Up</h2>
    <form (ngSubmit)="onSubmit()" #registerForm="ngForm">
      <div class="form-group">
        <label for="name">Full Name</label>
        <input
          type="text"
          id="name"
          name="name"
          [(ngModel)]="name"
          required
          class="form-control"
          placeholder="Enter your full name"
        />
      </div>

      <div class="form-group">
        <label for="email">Email</label>
        <input
          type="email"
          id="email"
          name="email"
          [(ngModel)]="email"
          required
          class="form-control"
          placeholder="Enter your email"
        />
      </div>

      <div class="form-group">
        <label for="password">Password</label>
        <input
          type="password"
          id="password"
          name="password"
          [(ngModel)]="password"
          required
          minlength="6"
          class="form-control"
          placeholder="Enter your password"
        />
      </div>

      <div class="form-group">
        <label for="confirmPassword">Confirm Password</label>
        <input
          type="password"
          id="confirmPassword"
          name="confirmPassword"
          [(ngModel)]="confirmPassword"
          required
          class="form-control"
          placeholder="Confirm your password"
        />
      </div>

      <div *ngIf="error" class="error-message">
        {{ error }}
      </div>

      <button
        type="submit"
        class="btn btn-primary"
        [disabled]="loading || !registerForm.valid"
      >
        {{ loading ? 'Creating account...' : 'Sign Up' }}
      </button>
    </form>

    <p class="auth-link">
      Already have an account? <a routerLink="/login">Login</a>
    </p>
  </div>
</div>


.auth-container {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 20px;
}

.auth-card {
  background: white;
  border-radius: 12px;
  padding: 40px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
  width: 100%;
  max-width: 400px;
}

.auth-card h2 {
  margin: 0 0 30px 0;
  color: #333;
  text-align: center;
}

.form-group {
  margin-bottom: 20px;
}

.form-group label {
  display: block;
  margin-bottom: 8px;
  color: #555;
  font-weight: 500;
}

.form-control {
  width: 100%;
  padding: 12px;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 14px;
  transition: border-color 0.3s;
  box-sizing: border-box;
}

.form-control:focus {
  outline: none;
  border-color: #667eea;
}

.btn {
  width: 100%;
  padding: 12px;
  border: none;
  border-radius: 6px;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
  transition: background-color 0.3s;
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

.error-message {
  color: #e74c3c;
  margin-bottom: 15px;
  padding: 10px;
  background: #fee;
  border-radius: 6px;
  font-size: 14px;
}

.auth-link {
  text-align: center;
  margin-top: 20px;
  color: #666;
}

.auth-link a {
  color: #667eea;
  text-decoration: none;
  font-weight: 500;
}

.auth-link a:hover {
  text-decoration: underline;
}


import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router, RouterModule } from '@angular/router';
import { AuthService } from '../../../services/auth.service';

@Component({
  selector: 'app-register',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterModule],
  templateUrl: './register.component.html',
  styleUrls: ['./register.component.css']
})
export class RegisterComponent {
  name: string = '';
  email: string = '';
  password: string = '';
  confirmPassword: string = '';
  error: string = '';
  loading: boolean = false;

  constructor(
    private authService: AuthService,
    private router: Router
  ) {}

  onSubmit() {
    if (!this.name || !this.email || !this.password || !this.confirmPassword) {
      this.error = 'Please fill in all fields';
      return;
    }

    if (this.password !== this.confirmPassword) {
      this.error = 'Passwords do not match';
      return;
    }

    if (this.password.length < 6) {
      this.error = 'Password must be at least 6 characters';
      return;
    }

    this.loading = true;
    this.error = '';

    this.authService.signup({
      name: this.name,
      email: this.email,
      password: this.password
    }).subscribe({
      next: () => {
        this.router.navigate(['/dashboard']);
      },
      error: (err) => {
        this.error = err.error?.detail || 'Registration failed. Please try again.';
        this.loading = false;
      }
    });
  }
}

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


.dashboard-container {
  min-height: 100vh;
  background: #f5f7fa;
}

.dashboard-header {
  background: white;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  padding: 20px 0;
  margin-bottom: 30px;
}

.header-content {
  max-width: 1200px;
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

.user-info {
  display: flex;
  align-items: center;
  gap: 15px;
}

.dashboard-content {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 20px;
}

.status-section {
  margin-bottom: 30px;
}

.status-section h2 {
  margin-bottom: 20px;
  color: #333;
}

.status-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
}

.status-card {
  background: white;
  border-radius: 8px;
  padding: 20px;
  display: flex;
  align-items: center;
  gap: 15px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.status-card.active {
  border-left: 4px solid #27ae60;
}

.status-card.idle {
  border-left: 4px solid #f39c12;
}

.status-card.error {
  border-left: 4px solid #e74c3c;
}

.status-icon {
  font-size: 24px;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  background: #f0f0f0;
}

.status-info h3 {
  margin: 0;
  font-size: 24px;
  color: #333;
}

.status-info p {
  margin: 5px 0 0 0;
  color: #666;
  font-size: 14px;
}

.logs-section {
  background: white;
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 30px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.logs-section h2 {
  margin-top: 0;
  margin-bottom: 15px;
  color: #333;
}

.logs-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.log-item {
  padding: 10px;
  border-radius: 4px;
  display: flex;
  gap: 15px;
  font-size: 14px;
}

.log-item.log-success {
  background: #d4edda;
  color: #155724;
}

.log-item.log-info {
  background: #d1ecf1;
  color: #0c5460;
}

.log-item.log-error {
  background: #f8d7da;
  color: #721c24;
}

.log-time {
  font-weight: 500;
  min-width: 150px;
}

.jobs-section {
  background: white;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.section-header h2 {
  margin: 0;
  color: #333;
}

.filters-bar {
  display: flex;
  gap: 20px;
  margin-bottom: 20px;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 6px;
  flex-wrap: wrap;
}

.filter-group {
  display: flex;
  align-items: center;
  gap: 10px;
}

.filter-group label {
  font-weight: 500;
  color: #555;
}

.filter-group select {
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
}

.jobs-table-container {
  overflow-x: auto;
}

.jobs-table {
  width: 100%;
  border-collapse: collapse;
}

.jobs-table th {
  background: #f8f9fa;
  padding: 12px;
  text-align: left;
  font-weight: 600;
  color: #555;
  border-bottom: 2px solid #dee2e6;
}

.jobs-table td {
  padding: 12px;
  border-bottom: 1px solid #dee2e6;
}

.badge {
  padding: 4px 12px;
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

.progress-bar {
  width: 100px;
  height: 8px;
  background: #e9ecef;
  border-radius: 4px;
  overflow: hidden;
  display: inline-block;
  margin-right: 10px;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
  transition: width 0.3s;
}

.progress-text {
  font-size: 12px;
  color: #666;
}

.btn-link {
  color: #667eea;
  text-decoration: none;
  margin-right: 10px;
  font-size: 14px;
}

.btn-link:hover {
  text-decoration: underline;
}

.btn {
  padding: 10px 20px;
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

.btn-secondary {
  background: #6c757d;
  color: white;
}

.btn-secondary:hover {
  background: #5a6268;
}

.btn-sm {
  padding: 6px 12px;
  font-size: 12px;
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.loading, .empty-state {
  text-align: center;
  padding: 40px;
  color: #666;
}

.empty-state a {
  color: #667eea;
  text-decoration: none;
}

.pagination {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 15px;
  margin-top: 20px;
  padding-top: 20px;
  border-top: 1px solid #dee2e6;
}


<div class="dashboard-container">
  <header class="dashboard-header">
    <div class="header-content">
      <h1>Dashboard</h1>
      <div class="user-info">
        <span>Welcome, {{ currentUser?.name || 'User' }}</span>
        <button class="btn btn-secondary" (click)="logout()">Logout</button>
      </div>
    </div>
  </header>

  <div class="dashboard-content">
    <!-- Agent Status Section -->
    <section class="status-section">
      <h2>Agent Status</h2>
      <div class="status-cards">
        <div class="status-card active">
          <div class="status-icon">✓</div>
          <div class="status-info">
            <h3>{{ agentStatus.active }}</h3>
            <p>Active Agents</p>
          </div>
        </div>
        <div class="status-card idle">
          <div class="status-icon">○</div>
          <div class="status-info">
            <h3>{{ agentStatus.idle }}</h3>
            <p>Idle Agents</p>
          </div>
        </div>
        <div class="status-card error">
          <div class="status-icon">✗</div>
          <div class="status-info">
            <h3>{{ agentStatus.error }}</h3>
            <p>Error Agents</p>
          </div>
        </div>
      </div>
    </section>

    <!-- Recent Logs -->
    <section class="logs-section">
      <h2>Recent Activity</h2>
      <div class="logs-list">
        <div *ngFor="let log of recentLogs" class="log-item" [ngClass]="'log-' + log.type">
          <span class="log-time">{{ log.timestamp | date:'short' }}</span>
          <span class="log-message">{{ log.message }}</span>
        </div>
      </div>
    </section>

    <!-- Jobs Section -->
    <section class="jobs-section">
      <div class="section-header">
        <h2>Research Jobs</h2>
        <a routerLink="/jobs/create" class="btn btn-primary">Create New Job</a>
      </div>

      <!-- Filters and Sorting -->
      <div class="filters-bar">
        <div class="filter-group">
          <label for="statusFilter">Status:</label>
          <select id="statusFilter" [(ngModel)]="statusFilter" (change)="onStatusFilterChange()">
            <option value="all">All</option>
            <option value="pending">Pending</option>
            <option value="running">Running</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
          </select>
        </div>

        <div class="filter-group">
          <label for="sortBy">Sort By:</label>
          <select id="sortBy" [(ngModel)]="sortBy" (change)="onSortChange()">
            <option value="created_at">Date</option>
            <option value="status">Status</option>
            <option value="progress">Progress</option>
          </select>
        </div>

        <div class="filter-group">
          <label for="sortOrder">Order:</label>
          <select id="sortOrder" [(ngModel)]="sortOrder" (change)="onSortChange()">
            <option value="desc">Descending</option>
            <option value="asc">Ascending</option>
          </select>
        </div>
      </div>

      <!-- Jobs Table -->
      <div class="jobs-table-container">
        <table class="jobs-table" *ngIf="!loading">
          <thead>
            <tr>
              <th>ID</th>
              <th>Type</th>
              <th>Status</th>
              <th>Progress</th>
              <th>Created</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            <tr *ngFor="let job of filteredJobs">
              <td>{{ job.id }}</td>
              <td>{{ job.type }}</td>
              <td>
                <span class="badge" [ngClass]="getStatusClass(job.status)">
                  {{ job.status }}
                </span>
              </td>
              <td>
                <div class="progress-bar">
                  <div class="progress-fill" [style.width.%]="job.progress"></div>
                </div>
                <span class="progress-text">{{ job.progress }}%</span>
              </td>
              <td>{{ job.created_at | date:'short' }}</td>
              <td>
                <a [routerLink]="['/jobs', job.id]" class="btn-link">View</a>
                <a *ngIf="job.status === 'running'" [routerLink]="['/jobs', job.id, 'progress']" class="btn-link">Progress</a>
              </td>
            </tr>
          </tbody>
        </table>

        <div *ngIf="loading" class="loading">Loading jobs...</div>
        <div *ngIf="!loading && filteredJobs.length === 0" class="empty-state">
          No jobs found. <a routerLink="/jobs/create">Create your first job</a>
        </div>
      </div>

      <!-- Pagination -->
      <div class="pagination" *ngIf="totalPages > 1">
        <button 
          class="btn btn-sm" 
          [disabled]="currentPage === 1"
          (click)="onPageChange(currentPage - 1)"
        >
          Previous
        </button>
        <span>Page {{ currentPage }} of {{ totalPages }}</span>
        <button 
          class="btn btn-sm" 
          [disabled]="currentPage === totalPages"
          (click)="onPageChange(currentPage + 1)"
        >
          Next
        </button>
      </div>
    </section>
  </div>
</div>


import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { ApiService, Job } from '../../services/api.service';
import { AuthService } from '../../services/auth.service';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule, RouterModule, FormsModule],
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.css']
})
export class DashboardComponent implements OnInit {
  jobs: Job[] = [];
  filteredJobs: Job[] = [];
  currentUser: any = null;
  loading: boolean = false;
  currentPage: number = 1;
  pageSize: number = 10;
  totalJobs: number = 0;

  // Filters
  statusFilter: string = 'all';
  sortBy: string = 'created_at';
  sortOrder: 'asc' | 'desc' = 'desc';

  // Agent status (mock data for now)
  agentStatus = {
    active: 2,
    idle: 1,
    error: 0
  };

  recentLogs = [
    { timestamp: new Date(), message: 'Job #123 completed successfully', type: 'success' },
    { timestamp: new Date(), message: 'Job #124 started processing', type: 'info' }
  ];

  constructor(
    private apiService: ApiService,
    private authService: AuthService
  ) {}

  ngOnInit() {
    this.authService.currentUser$.subscribe(user => {
      this.currentUser = user;
    });
    this.loadJobs();
  }

  loadJobs() {
    this.loading = true;
    this.apiService.getJobs({
      page: this.currentPage,
      limit: this.pageSize,
      status: this.statusFilter !== 'all' ? this.statusFilter : undefined
    }).subscribe({
      next: (response) => {
        this.jobs = response.jobs;
        this.totalJobs = response.total;
        this.applyFilters();
        this.loading = false;
      },
      error: (err) => {
        console.error('Error loading jobs:', err);
        this.loading = false;
      }
    });
  }

  applyFilters() {
    this.filteredJobs = [...this.jobs];
    
    if (this.statusFilter !== 'all') {
      this.filteredJobs = this.filteredJobs.filter(job => job.status === this.statusFilter);
    }

    this.filteredJobs.sort((a, b) => {
      let aVal: any, bVal: any;
      
      switch (this.sortBy) {
        case 'created_at':
          aVal = new Date(a.created_at).getTime();
          bVal = new Date(b.created_at).getTime();
          break;
        case 'status':
          aVal = a.status;
          bVal = b.status;
          break;
        case 'progress':
          aVal = a.progress;
          bVal = b.progress;
          break;
        default:
          aVal = a.id;
          bVal = b.id;
      }

      if (this.sortOrder === 'asc') {
        return aVal > bVal ? 1 : -1;
      } else {
        return aVal < bVal ? 1 : -1;
      }
    });
  }

  onStatusFilterChange() {
    this.loadJobs();
  }

  onSortChange() {
    this.applyFilters();
  }

  getStatusClass(status: string): string {
    return `status-${status}`;
  }

  onPageChange(page: number) {
    this.currentPage = page;
    this.loadJobs();
  }

  get totalPages(): number {
    return Math.ceil(this.totalJobs / this.pageSize);
  }

  logout() {
    this.authService.logout();
  }
}

import { HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';
import { AuthService } from '../services/auth.service';

export const authInterceptor: HttpInterceptorFn = (req, next) => {
  const authService = inject(AuthService);
  const token = authService.getToken();

  if (token) {
    const cloned = req.clone({
      setHeaders: {
        Authorization: `Bearer ${token}`
      }
    });
    return next(cloned);
  }

  return next(req);
};



import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, BehaviorSubject, tap } from 'rxjs';
import { environment } from '../../environments/environment';

export interface LoginRequest {
  username: string; // email
  password: string;
}

export interface SignupRequest {
  email: string;
  password: string;
  name: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
}

export interface User {
  id: number;
  email: string;
  name: string;
}

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private tokenKey = 'auth_token';
  private currentUserSubject = new BehaviorSubject<User | null>(null);
  public currentUser$ = this.currentUserSubject.asObservable();

  constructor(private http: HttpClient) {
    this.loadUserFromToken();
  }

  login(credentials: LoginRequest): Observable<AuthResponse> {
    const formData = new FormData();
    formData.append('username', credentials.username);
    formData.append('password', credentials.password);

    return this.http.post<AuthResponse>(`${environment.apiBaseUrl}/auth/login`, formData).pipe(
      tap(response => {
        this.setToken(response.access_token);
        this.loadUserFromToken();
      })
    );
  }

  signup(userData: SignupRequest): Observable<AuthResponse> {
    return this.http.post<AuthResponse>(`${environment.apiBaseUrl}/auth/signup`, userData).pipe(
      tap(response => {
        this.setToken(response.access_token);
        this.loadUserFromToken();
      })
    );
  }

  logout(): void {
    localStorage.removeItem(this.tokenKey);
    this.currentUserSubject.next(null);
  }

  getToken(): string | null {
    return localStorage.getItem(this.tokenKey);
  }

  isAuthenticated(): boolean {
    return !!this.getToken();
  }

  private setToken(token: string): void {
    localStorage.setItem(this.tokenKey, token);
  }

  private loadUserFromToken(): void {
    const token = this.getToken();
    if (token) {
      // Decode JWT to get user info (simple decode, not verified)
      try {
        const payload = JSON.parse(atob(token.split('.')[1]));
        this.currentUserSubject.next({
          id: 0, // Will be fetched from backend if needed
          email: payload.sub,
          name: payload.sub.split('@')[0]
        });
      } catch (e) {
        console.error('Error decoding token', e);
      }
    }
  }

  getAuthHeaders(): { [key: string]: string } {
    const token = this.getToken();
    return token ? { 'Authorization': `Bearer ${token}` } : {};
  }
}

import { ApplicationConfig, provideBrowserGlobalErrorListeners } from '@angular/core';
import { provideRouter } from '@angular/router';
import { provideHttpClient, withInterceptors } from '@angular/common/http';

import { routes } from './app.routes';
import { authInterceptor } from './interceptors/auth.interceptor';

export const appConfig: ApplicationConfig = {
  providers: [
    provideBrowserGlobalErrorListeners(),
    provideRouter(routes),
    provideHttpClient(withInterceptors([authInterceptor]))
  ]
};


import { Routes } from '@angular/router';
import { authGuard } from './guards/auth.guard';
import { adminGuard } from './guards/admin.guard';

export const routes: Routes = [
  {
    path: '',
    redirectTo: '/dashboard',
    pathMatch: 'full'
  },
  {
    path: 'login',
    loadComponent: () => import('./components/auth/login/login.component').then(m => m.LoginComponent)
  },
  {
    path: 'register',
    loadComponent: () => import('./components/auth/register/register.component').then(m => m.RegisterComponent)
  },
  {
    path: 'dashboard',
    loadComponent: () => import('./components/dashboard/dashboard.component').then(m => m.DashboardComponent),
    canActivate: [authGuard]
  },
  {
    path: 'jobs/create',
    loadComponent: () => import('./components/jobs/create-job/create-job.component').then(m => m.CreateJobComponent),
    canActivate: [authGuard]
  },
  {
    path: 'jobs/:id',
    loadComponent: () => import('./components/jobs/progress/progress.component').then(m => m.ProgressComponent),
    canActivate: [authGuard]
  },
  {
    path: 'jobs/:id/progress',
    loadComponent: () => import('./components/jobs/progress/progress.component').then(m => m.ProgressComponent),
    canActivate: [authGuard]
  },
  {
    path: 'reports/:id',
    loadComponent: () => import('./components/reports/report-view/report-view.component').then(m => m.ReportViewComponent),
    canActivate: [authGuard]
  },
  {
    path: 'reports/:id/edit',
    loadComponent: () => import('./components/reports/report-edit/report-edit.component').then(m => m.ReportEditComponent),
    canActivate: [authGuard]
  },
  {
    path: 'admin',
    loadComponent: () => import('./components/admin/admin-panel/admin-panel.component').then(m => m.AdminPanelComponent),
    canActivate: [adminGuard]
  },
  {
    path: '**',
    redirectTo: '/dashboard'
  }
];
