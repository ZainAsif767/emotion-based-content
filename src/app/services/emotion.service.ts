import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import * as csv from 'csv-parser';
import { catchError, delay, Observable, of, Subject, throwError } from 'rxjs';
import { Papa } from 'ngx-papaparse';
import { environment } from '../../environment/environment';

@Injectable({
  providedIn: 'root'
})
export class EmotionService {
  private emotionSubject = new Subject<string>();
  emotion$ = this.emotionSubject.asObservable();
  private recommendations: { [key: string]: string[] } = {};
  papa = new Papa();

  constructor(private http: HttpClient) {
    this.loadCSVData();
  }

  private loadCSVData() {
    this.http.get('assets/emotions.csv', { responseType: 'text' })
      .subscribe(data => {
        this.papa.parse(data, {
          header: true,
          complete: (result: any) => {
            result.data.forEach((row: any) => {
              const emotion = row['emotion'];
              const content = row['content'];
              if (!this.recommendations[emotion]) {
                this.recommendations[emotion] = [];
              }
              this.recommendations[emotion].push(content);
            });
          }
        });
      });
  }


  analyzeText(text: string): Observable<any> {
    return this.http.post<any>(`${environment.apiUrl}/analyze`, { text })
      .pipe(
        catchError(error => {
          console.error('API Error:', error);
          return throwError(() => new Error('Failed to analyze text'));
        })
      );
  }

  analyzeEmotion(text: string) {
    // this.http.post<{ emotion: string }>('http://localhost:5000/analyze', { text })
    //   .subscribe(response => {
    //     const emotion = response.emotion;
    //     this.emotionSubject.next(emotion);
    //   });

    // Mock response for development
    console.log(text)
    const mockResponse = { emotion: this.mockEmotionDetection(text) };
    of(mockResponse).pipe(delay(500)).subscribe((response: any) => {
      const emotion = response.emotion;
      this.emotionSubject.next(emotion);
    });
  }

  private mockEmotionDetection(text: string): string {
    // Simple mock logic to detect emotion based on keywords
    console.log(text)
    if (text.includes('happy')) {
      return 'happy';
    } else if (text.includes('sad')) {
      return 'sad';
    } else if (text.includes('angry')) {
      return 'angry';
    } else {
      return 'neutral';
    }
  }

  getRecommendations(emotion: string): string[] {
    return this.recommendations[emotion] || [];
  }
}