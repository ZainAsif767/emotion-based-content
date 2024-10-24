import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { EmotionService } from '../../services/emotion.service';

@Component({
  selector: 'app-recommendations',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './recommendations.component.html',
  styleUrl: './recommendations.component.scss'
})
export class RecommendationsComponent {
  recommendations: string[] = [];

  constructor(private emotionService: EmotionService) { }

  ngOnInit() {
    this.emotionService.emotion$.subscribe(emotion => {
      console.log('Detected emotion:', emotion); // Debugging line
      this.recommendations = this.emotionService.getRecommendations(emotion);
      console.log('Recommendations:', this.recommendations); // Debugging line
    });
  }
}
