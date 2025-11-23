<div class="dashboard-container">
  <header class="dashboard-header">
    <h1>Research Dashboard</h1>
    <div class="user-info">
      <span>Welcome, {{ (authService.currentUser$ | async)?.name }}</span>
      <button (click)="authService.logout()" class="btn btn-outline">Logout</button>
    </div>
  </header>

  <div class="actions-bar">
    <a routerLink="/jobs/create" class="btn btn-primary">+ New Research Job</a>
  </div>

  <div class="jobs-list-container">
    <h2>Recent Jobs</h2>

    <div *ngIf="loading" class="loading-spinner">Loading jobs...</div>

    <div *ngIf="error" class="error-message">{{ error }}</div>

    <div *ngIf="!loading && jobs.length === 0" class="empty-state">
      <p>No jobs found. Start a new research task!</p>
    </div>

    <table *ngIf="!loading && jobs.length > 0" class="jobs-table">
      <thead>
        <tr>
          <th>ID</th>
          <th>Name</th>
          <th>Type</th>
          <th>Status</th>
          <th>Created</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody>
        <tr *ngFor="let job of jobs">
          <td>#{{ job.id }}</td>
          <td>{{ job.name || 'Untitled Job' }}</td>
          <td>{{ job.type }}</td>
          <td>
            <span class="status-badge" [ngClass]="getStatusClass(job.status)">
              {{ job.status }}
            </span>
          </td>
          <td>{{ job.created_at | date:'short' }}</td>
          <td>
            <a [routerLink]="['/jobs', job.id]" class="btn btn-sm btn-secondary">View</a>
          </td>
        </tr>
      </tbody>
    </table>

    <div class="pagination-controls" *ngIf="jobs.length > 0">
      <button (click)="prevPage()" [disabled]="currentPage === 1" class="btn btn-sm">Previous</button>
      <span class="page-info">Page {{ currentPage }}</span>
      <button (click)="nextPage()" [disabled]="!hasMore" class="btn btn-sm">Next</button>
    </div>
  </div>
</div>


.dashboard-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
}

.dashboard-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid #eee;
}

.user-info {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.actions-bar {
  margin-bottom: 2rem;
}

.jobs-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 1rem;
  background: white;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  border-radius: 8px;
  overflow: hidden;
}

.jobs-table th,
.jobs-table td {
  padding: 1rem;
  text-align: left;
  border-bottom: 1px solid #eee;
}

.jobs-table th {
  background-color: #f8f9fa;
  font-weight: 600;
  color: #444;
}

.status-badge {
  padding: 0.25rem 0.75rem;
  border-radius: 999px;
  font-size: 0.875rem;
  font-weight: 500;
  text-transform: capitalize;
}

.status-completed {
  background-color: #d1fae5;
  color: #065f46;
}

.status-running {
  background-color: #dbeafe;
  color: #1e40af;
}

.status-failed {
  background-color: #fee2e2;
  color: #991b1b;
}

.status-pending {
  background-color: #f3f4f6;
  color: #374151;
}

.pagination-controls {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 1rem;
  margin-top: 2rem;
}

.btn-sm {
  padding: 0.25rem 0.75rem;
  font-size: 0.875rem;
}

.empty-state {
  text-align: center;
  padding: 3rem;
  color: #666;
  background: #f9fafb;
  border-radius: 8px;
}


import { Component, OnInit, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { JobService, Job } from '../../services/job.service';
import { AuthService } from '../../services/auth.service';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.css']
})
export class DashboardComponent implements OnInit {
  private jobService = inject(JobService);
  public authService = inject(AuthService);

  jobs: Job[] = [];
  loading: boolean = true;
  error: string = '';

  // Pagination
  currentPage: number = 1;
  pageSize: number = 10;
  hasMore: boolean = false; // Simple check, ideally backend returns total count

  ngOnInit(): void {
    this.loadJobs();
  }

  loadJobs(): void {
    this.loading = true;
    this.jobService.getJobs(this.currentPage, this.pageSize).subscribe({
      next: (data) => {
        this.jobs = data;
        this.loading = false;
        // Heuristic for pagination if backend doesn't return count
        this.hasMore = data.length === this.pageSize;
      },
      error: (err) => {
        this.error = 'Failed to load jobs.';
        this.loading = false;
        console.error(err);
      }
    });
  }

  nextPage(): void {
    if (this.hasMore) {
      this.currentPage++;
      this.loadJobs();
    }
  }

  prevPage(): void {
    if (this.currentPage > 1) {
      this.currentPage--;
      this.loadJobs();
    }
  }

  getStatusClass(status: string): string {
    switch (status) {
      case 'completed': return 'status-completed';
      case 'failed': return 'status-failed';
      case 'running': return 'status-running';
      default: return 'status-pending';
    }
  }
}

