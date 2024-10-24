import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { MaterialModule } from '../../../material.module';
import { EmotionService } from '../../services/emotion.service';

@Component({
  selector: 'app-input-form',
  standalone: true,
  imports: [ReactiveFormsModule, CommonModule, MaterialModule, FormsModule],
  templateUrl: './input-form.component.html',
  styleUrl: './input-form.component.scss'
})
export class InputFormComponent {
  userInput: string = '';
  recommendations: string[] = [];

  private mockRecommendations = {
    happy: ['Watch a comedy movie', 'Read a funny book'],
    sad: ['Listen to soothing music', 'Watch a motivational video'],
    angry: ['Practice deep breathing', 'Go for a walk'],
    neutral: ['Read a book', 'Watch a documentary']
  };


  constructor(private emotionService: EmotionService) {
    this.emotionService.emotion$.subscribe((value => {
      console.log(value)
    }))
  }

  ngOnInit() {
    this.emotionService.emotion$.subscribe(emotion => {
      console.log('Detected emotion:', emotion); // Debugging line
      this.recommendations = this.emotionService.getRecommendations(emotion);
      console.log('Recommendations:', this.recommendations); // Debugging line
    });
  }

  onSubmit() {
    console.log('User input:', this.userInput); // Debugging line
    this.emotionService.analyzeEmotion(this.userInput);

  }
}
