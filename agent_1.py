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
