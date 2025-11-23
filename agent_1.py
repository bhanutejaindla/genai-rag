<div class="auth-container">
  <div class="auth-card">
    <h2>Sign Up</h2>
    <form (ngSubmit)="onSubmit()" #registerForm="ngForm">
      <div class="form-group">
        <label for="username">Username *</label>
        <input
          type="text"
          id="username"
          name="username"
          [(ngModel)]="username"
          required
          minlength="3"
          pattern="[a-zA-Z0-9_]+"
          class="form-control"
          [ngClass]="{'error': hasFieldError('username')}"
          placeholder="Enter your username"
        />
        <div *ngIf="hasFieldError('username')" class="field-error">
          {{ getFieldError('username') }}
        </div>
        <div class="field-hint">3+ characters, letters, numbers, and underscores only</div>
      </div>

      <div class="form-group">
        <label for="email">Email *</label>
        <input
          type="email"
          id="email"
          name="email"
          [(ngModel)]="email"
          required
          class="form-control"
          [ngClass]="{'error': hasFieldError('email')}"
          placeholder="Enter your email"
        />
        <div *ngIf="hasFieldError('email')" class="field-error">
          {{ getFieldError('email') }}
        </div>
      </div>

      <div class="form-group">
        <label for="password">Password *</label>
        <input
          type="password"
          id="password"
          name="password"
          [(ngModel)]="password"
          required
          minlength="6"
          maxlength="128"
          class="form-control"
          [ngClass]="{'error': hasFieldError('password')}"
          placeholder="Enter your password"
        />
        <div *ngIf="hasFieldError('password')" class="field-error">
          {{ getFieldError('password') }}
        </div>
        <div class="field-hint">6+ characters, must include uppercase, lowercase, and number</div>
      </div>

      <div class="form-group">
        <label for="confirmPassword">Confirm Password *</label>
        <input
          type="password"
          id="confirmPassword"
          name="confirmPassword"
          [(ngModel)]="confirmPassword"
          required
          class="form-control"
          [ngClass]="{'error': hasFieldError('confirmPassword')}"
          placeholder="Confirm your password"
        />
        <div *ngIf="hasFieldError('confirmPassword')" class="field-error">
          {{ getFieldError('confirmPassword') }}
        </div>
      </div>

      <div class="form-group">
        <label for="role">Role *</label>
        <select
          id="role"
          name="role"
          [(ngModel)]="role"
          required
          class="form-control"
          [ngClass]="{'error': hasFieldError('role')}"
        >
          <option value="user">User</option>
          <option value="admin">Admin</option>
        </select>
        <div *ngIf="hasFieldError('role')" class="field-error">
          {{ getFieldError('role') }}
        </div>
        <div class="field-hint">Select your account role</div>
      </div>

      <div *ngIf="errors['submit']" class="error-message">
        {{ errors['submit'] }}
      </div>

      <button
        type="submit"
        class="btn btn-primary"
        [disabled]="loading"
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

.form-control.error {
  border-color: #e74c3c;
}

.form-control.error:focus {
  border-color: #e74c3c;
}

.field-error {
  color: #e74c3c;
  font-size: 12px;
  margin-top: 5px;
  display: block;
}

.field-hint {
  color: #666;
  font-size: 11px;
  margin-top: 4px;
  display: block;
  font-style: italic;
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

export type UserRole = 'user' | 'admin';

@Component({
  selector: 'app-register',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterModule],
  templateUrl: './register.component.html',
  styleUrls: ['./register.component.css']
})
export class RegisterComponent {
  username: string = '';
  email: string = '';
  password: string = '';
  confirmPassword: string = '';
  role: UserRole = 'user';
  errors: { [key: string]: string } = {};
  loading: boolean = false;

  constructor(
    private authService: AuthService,
    private router: Router
  ) {}

  validate(): boolean {
    this.errors = {};

    // Username validation
    if (!this.username || this.username.trim().length === 0) {
      this.errors['username'] = 'Username is required';
    } else if (this.username.trim().length < 3) {
      this.errors['username'] = 'Username must be at least 3 characters';
    } else if (!/^[a-zA-Z0-9_]+$/.test(this.username.trim())) {
      this.errors['username'] = 'Username can only contain letters, numbers, and underscores';
    }

    // Email validation
    if (!this.email || this.email.trim().length === 0) {
      this.errors['email'] = 'Email is required';
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(this.email.trim())) {
      this.errors['email'] = 'Please enter a valid email address';
    }

    // Password validation
    if (!this.password) {
      this.errors['password'] = 'Password is required';
    } else if (this.password.length < 6) {
      this.errors['password'] = 'Password must be at least 6 characters';
    } else if (this.password.length > 128) {
      this.errors['password'] = 'Password must be less than 128 characters';
    } else if (!/(?=.*[a-z])/.test(this.password)) {
      this.errors['password'] = 'Password must contain at least one lowercase letter';
    } else if (!/(?=.*[A-Z])/.test(this.password)) {
      this.errors['password'] = 'Password must contain at least one uppercase letter';
    } else if (!/(?=.*\d)/.test(this.password)) {
      this.errors['password'] = 'Password must contain at least one number';
    }

    // Confirm password validation
    if (!this.confirmPassword) {
      this.errors['confirmPassword'] = 'Please confirm your password';
    } else if (this.password !== this.confirmPassword) {
      this.errors['confirmPassword'] = 'Passwords do not match';
    }

    // Role validation
    if (!this.role || (this.role !== 'user' && this.role !== 'admin')) {
      this.errors['role'] = 'Please select a valid role';
    }

    return Object.keys(this.errors).length === 0;
  }

  onSubmit() {
    if (!this.validate()) {
      return;
    }

    this.loading = true;
    this.errors = {};

    this.authService.signup({
      username: this.username.trim(),
      email: this.email.trim(),
      password: this.password,
      role: this.role
    }).subscribe({
      next: () => {
        this.router.navigate(['/dashboard']);
      },
      error: (err) => {
        this.errors['submit'] = err.error?.detail || 'Registration failed. Please try again.';
        this.loading = false;
      }
    });
  }

  getFieldError(fieldName: string): string {
    return this.errors[fieldName] || '';
  }

  hasFieldError(fieldName: string): boolean {
    return !!this.errors[fieldName];
  }
}



