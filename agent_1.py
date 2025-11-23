import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, BehaviorSubject, tap } from 'rxjs';
import { environment } from '../../environments/environment';

export interface LoginRequest {
  email: string;
  password: string;
}

export interface SignupRequest {
  username: string;
  email: string;
  password: string;
  role: 'user' | 'admin';
}

export interface AuthResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

export interface User {
  id: number;
  username: string;
  email: string;
  name: string;
  role?: 'user' | 'admin';
}

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private http = inject(HttpClient);

  private accessKey = 'access_token';
  private refreshKey = 'refresh_token';

  private currentUserSubject = new BehaviorSubject<User | null>(null);
  public currentUser$ = this.currentUserSubject.asObservable();

  constructor() {
    this.loadUserFromToken();
  }

  // ---------------------------
  // AUTH API CALLS
  // ---------------------------

  login(credentials: LoginRequest): Observable<AuthResponse> {
    const formData = new FormData();
    formData.append('username', credentials.email); // FastAPI expects "username"
    formData.append('password', credentials.password);

    return this.http.post<AuthResponse>(`${environment.apiBaseUrl}/auth/login`, formData).pipe(
      tap(res => {
        this.setTokens(res.access_token, res.refresh_token);
        this.loadUserFromToken();
      })
    );
  }

  signup(data: SignupRequest): Observable<AuthResponse> {
    return this.http.post<AuthResponse>(`${environment.apiBaseUrl}/auth/signup`, data).pipe(
      tap(res => {
        this.setTokens(res.access_token, res.refresh_token);
        this.loadUserFromToken();
      })
    );
  }

  // ---------------------------
  // TOKEN HANDLING
  // ---------------------------

  private setTokens(access: string, refresh: string): void {
    localStorage.setItem(this.accessKey, access);
    localStorage.setItem(this.refreshKey, refresh);
  }

  getAccessToken(): string | null {
    return localStorage.getItem(this.accessKey);
  }

  getRefreshToken(): string | null {
    return localStorage.getItem(this.refreshKey);
  }

  logout(): void {
    localStorage.removeItem(this.accessKey);
    localStorage.removeItem(this.refreshKey);
    this.currentUserSubject.next(null);
  }

  isAuthenticated(): boolean {
    return !!this.getAccessToken();
  }

  // ---------------------------
  // DECODE USER FROM JWT
  // ---------------------------

  private loadUserFromToken(): void {
    const token = this.getAccessToken();
    if (!token) {
      this.currentUserSubject.next(null);
      return;
    }

    try {
      const payload = JSON.parse(atob(token.split('.')[1]));

      this.currentUserSubject.next({
        id: 0,
        username: payload.username || payload.sub?.split('@')[0] || 'user',
        email: payload.sub,
        name: payload.username || payload.sub?.split('@')[0],
        role: payload.role || 'user'
      });

    } catch (e) {
      console.error('Invalid JWT token:', e);
      this.currentUserSubject.next(null);
    }
  }

  getUserRole(): 'user' | 'admin' | null {
    return this.currentUserSubject.value?.role || null;
  }

  isAdmin(): boolean {
    return this.getUserRole() === 'admin';
  }

  getAuthHeaders(): { [key: string]: string } {
    const token = this.getAccessToken();
    return token ? { Authorization: `Bearer ${token}` } : {};
  }
}
