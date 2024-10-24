import { Routes } from '@angular/router';
import { provideRouter } from '@angular/router';
import { InputFormComponent } from './components/input-form/input-form.component';
import { RecommendationsComponent } from './components/recommendations/recommendations.component';

export const routes: Routes = [
    { path: '', redirectTo: '/input', pathMatch: 'full' },
    { path: 'input', component: InputFormComponent },
    { path: 'recommendations', component: RecommendationsComponent }
];

export const appRoutingProviders = [
    provideRouter(routes)
];