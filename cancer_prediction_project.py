#!/usr/bin/env python3
"""
Breast Cancer Prediction using Deep Learning with PyTorch
=========================================================

This project implements a deep neural network for breast cancer classification
using the Wisconsin Breast Cancer dataset. It includes comprehensive data 
analysis, visualization, preprocessing, and model training.

Author: AI Assistant
Dataset: Breast Cancer Wisconsin (Diagnostic)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CancerDataAnalyzer:
    """Class for data analysis and visualization"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load and prepare the dataset"""
        print("📊 Loading Cancer Dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"✅ Dataset loaded successfully!")
        print(f"📈 Shape: {self.df.shape}")
        print(f"🏷️  Columns: {list(self.df.columns)}")
        
    def basic_info(self):
        """Display basic information about the dataset"""
        print("\n" + "="*80)
        print("📋 DATASET OVERVIEW")
        print("="*80)
        
        print("\n🔍 Basic Information:")
        print(f"• Total samples: {len(self.df)}")
        print(f"• Total features: {len(self.df.columns)}")
        print(f"• Missing values: {self.df.isnull().sum().sum()}")
        
        print("\n📊 Target Distribution:")
        diagnosis_counts = self.df['diagnosis'].value_counts()
        print(f"• Malignant (M): {diagnosis_counts['M']} ({diagnosis_counts['M']/len(self.df)*100:.1f}%)")
        print(f"• Benign (B): {diagnosis_counts['B']} ({diagnosis_counts['B']/len(self.df)*100:.1f}%)")
        
        print(f"\n📈 Feature Statistics:")
        print(self.df.describe().round(3))
        
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n🎨 Creating Visualizations...")
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Target Distribution
        plt.subplot(4, 3, 1)
        diagnosis_counts = self.df['diagnosis'].value_counts()
        colors = ['#FF6B6B', '#4ECDC4']
        plt.pie(diagnosis_counts.values, labels=['Benign', 'Malignant'], colors=colors, 
                autopct='%1.1f%%', startangle=90)
        plt.title('🎯 Cancer Diagnosis Distribution', fontsize=14, fontweight='bold')
        
        # 2. Correlation Heatmap (top features)
        plt.subplot(4, 3, 2)
        # Select numerical columns excluding 'id'
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.drop('id', errors='ignore')
        corr_matrix = self.df[numeric_cols].corr()
        
        # Get top 10 features with highest correlation to create readable heatmap
        feature_cols = [col for col in numeric_cols if 'mean' in col][:10]
        if len(feature_cols) > 0:
            small_corr = self.df[feature_cols].corr()
            sns.heatmap(small_corr, annot=True, cmap='RdYlBu_r', center=0, 
                       fmt='.2f', square=True, cbar_kws={'shrink': 0.8})
        plt.title('🔥 Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # 3. Feature Distribution by Diagnosis
        plt.subplot(4, 3, 3)
        if 'radius_mean' in self.df.columns:
            for diagnosis in ['M', 'B']:
                data = self.df[self.df['diagnosis'] == diagnosis]['radius_mean']
                plt.hist(data, alpha=0.7, label=f'{"Malignant" if diagnosis == "M" else "Benign"}', 
                        bins=20, density=True)
            plt.xlabel('Radius Mean')
            plt.ylabel('Density')
            plt.title('📏 Radius Mean Distribution by Diagnosis', fontsize=14, fontweight='bold')
            plt.legend()
        
        # 4. Texture Mean Distribution
        plt.subplot(4, 3, 4)
        if 'texture_mean' in self.df.columns:
            sns.boxplot(data=self.df, x='diagnosis', y='texture_mean', palette=['#4ECDC4', '#FF6B6B'])
            plt.title('🔍 Texture Mean by Diagnosis', fontsize=14, fontweight='bold')
            plt.xlabel('Diagnosis')
            plt.ylabel('Texture Mean')
        
        # 5. Area vs Perimeter Scatter
        plt.subplot(4, 3, 5)
        if 'area_mean' in self.df.columns and 'perimeter_mean' in self.df.columns:
            for diagnosis, color in zip(['B', 'M'], ['#4ECDC4', '#FF6B6B']):
                mask = self.df['diagnosis'] == diagnosis
                plt.scatter(self.df[mask]['area_mean'], self.df[mask]['perimeter_mean'], 
                           c=color, alpha=0.6, s=50, 
                           label=f'{"Benign" if diagnosis == "B" else "Malignant"}')
            plt.xlabel('Area Mean')
            plt.ylabel('Perimeter Mean')
            plt.title('📐 Area vs Perimeter Mean', fontsize=14, fontweight='bold')
            plt.legend()
        
        # 6. Smoothness Distribution
        plt.subplot(4, 3, 6)
        if 'smoothness_mean' in self.df.columns:
            sns.violinplot(data=self.df, x='diagnosis', y='smoothness_mean', palette=['#4ECDC4', '#FF6B6B'])
            plt.title('🌊 Smoothness Mean Distribution', fontsize=14, fontweight='bold')
        
        # 7. Compactness vs Concavity
        plt.subplot(4, 3, 7)
        if 'compactness_mean' in self.df.columns and 'concavity_mean' in self.df.columns:
            for diagnosis, color in zip(['B', 'M'], ['#4ECDC4', '#FF6B6B']):
                mask = self.df['diagnosis'] == diagnosis
                plt.scatter(self.df[mask]['compactness_mean'], self.df[mask]['concavity_mean'], 
                           c=color, alpha=0.6, s=50,
                           label=f'{"Benign" if diagnosis == "B" else "Malignant"}')
            plt.xlabel('Compactness Mean')
            plt.ylabel('Concavity Mean')
            plt.title('🔄 Compactness vs Concavity', fontsize=14, fontweight='bold')
            plt.legend()
        
        # 8. Symmetry Distribution
        plt.subplot(4, 3, 8)
        if 'symmetry_mean' in self.df.columns:
            plt.hist([self.df[self.df['diagnosis'] == 'B']['symmetry_mean'],
                     self.df[self.df['diagnosis'] == 'M']['symmetry_mean']], 
                    bins=20, alpha=0.7, label=['Benign', 'Malignant'], 
                    color=['#4ECDC4', '#FF6B6B'])
            plt.xlabel('Symmetry Mean')
            plt.ylabel('Frequency')
            plt.title('⚖️ Symmetry Distribution', fontsize=14, fontweight='bold')
            plt.legend()
        
        # 9. Fractal Dimension
        plt.subplot(4, 3, 9)
        if 'fractal_dimension_mean' in self.df.columns:
            sns.stripplot(data=self.df, x='diagnosis', y='fractal_dimension_mean', 
                         palette=['#4ECDC4', '#FF6B6B'], size=6, alpha=0.7)
            plt.title('🔢 Fractal Dimension Distribution', fontsize=14, fontweight='bold')
        
        # 10. Feature Importance (Standard Deviation)
        plt.subplot(4, 3, 10)
        numeric_features = [col for col in self.df.columns if 'mean' in col][:10]
        if numeric_features:
            feature_std = self.df[numeric_features].std().sort_values(ascending=True)
            plt.barh(range(len(feature_std)), feature_std.values, color='skyblue')
            plt.yticks(range(len(feature_std)), [col.replace('_mean', '') for col in feature_std.index])
            plt.xlabel('Standard Deviation')
            plt.title('📊 Feature Variability', fontsize=14, fontweight='bold')
        
        # 11. Worst Features Comparison
        plt.subplot(4, 3, 11)
        worst_features = [col for col in self.df.columns if 'worst' in col][:5]
        if worst_features:
            malignant_means = self.df[self.df['diagnosis'] == 'M'][worst_features].mean()
            benign_means = self.df[self.df['diagnosis'] == 'B'][worst_features].mean()
            
            x = np.arange(len(worst_features))
            width = 0.35
            
            plt.bar(x - width/2, malignant_means, width, label='Malignant', color='#FF6B6B', alpha=0.8)
            plt.bar(x + width/2, benign_means, width, label='Benign', color='#4ECDC4', alpha=0.8)
            
            plt.xlabel('Features')
            plt.ylabel('Mean Values')
            plt.title('⚠️ Worst Features Comparison', fontsize=14, fontweight='bold')
            plt.xticks(x, [col.replace('_worst', '') for col in worst_features], rotation=45)
            plt.legend()
        
        # 12. SE Features Analysis
        plt.subplot(4, 3, 12)
        se_features = [col for col in self.df.columns if '_se' in col][:6]
        if se_features:
            # Create a box plot for SE features
            se_data = []
            labels = []
            for feature in se_features:
                se_data.extend([self.df[self.df['diagnosis'] == 'B'][feature].values,
                              self.df[self.df['diagnosis'] == 'M'][feature].values])
                labels.extend([f'B-{feature.replace("_se", "")}', f'M-{feature.replace("_se", "")}'])
            
            # Create simplified visualization
            malignant_se = self.df[self.df['diagnosis'] == 'M'][se_features].mean()
            benign_se = self.df[self.df['diagnosis'] == 'B'][se_features].mean()
            
            x = np.arange(len(se_features))
            plt.plot(x, malignant_se, 'o-', color='#FF6B6B', linewidth=2, markersize=8, label='Malignant')
            plt.plot(x, benign_se, 's-', color='#4ECDC4', linewidth=2, markersize=8, label='Benign')
            
            plt.xlabel('SE Features')
            plt.ylabel('Mean Values')
            plt.title('📈 Standard Error Features', fontsize=14, fontweight='bold')
            plt.xticks(x, [col.replace('_se', '') for col in se_features], rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cancer_data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved as 'cancer_data_analysis.png'")

class CancerNeuralNetwork(nn.Module):
    """Deep Neural Network for Cancer Classification"""
    
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.3):
        super(CancerNeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class CancerPredictor:
    """Main class for cancer prediction with deep learning"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {self.device}")
        
    def load_and_preprocess_data(self):
        """Load and preprocess the data"""
        print("\n" + "="*80)
        print("🔧 DATA PREPROCESSING")
        print("="*80)
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        print(f"✅ Data loaded: {self.df.shape}")
        
        # Remove ID column if present
        if 'id' in self.df.columns:
            self.df = self.df.drop('id', axis=1)
            print("🗑️  Removed ID column")
        
        # Separate features and target
        X = self.df.drop('diagnosis', axis=1)
        y = self.df['diagnosis']
        
        # Encode target variable (M=1, B=0)
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"📊 Feature matrix shape: {X.shape}")
        print(f"🎯 Target distribution after encoding: {np.bincount(y_encoded)}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"🔄 Data scaling completed")
        print(f"📈 Training set: {X_train_scaled.shape}")
        print(f"📉 Test set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def create_data_loaders(self, X_train, X_test, y_train, y_test, batch_size=32):
        """Create PyTorch data loaders"""
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_train_tensor = torch.FloatTensor(y_train)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def build_model(self, input_size):
        """Build the neural network model"""
        self.model = CancerNeuralNetwork(
            input_size=input_size,
            hidden_sizes=[128, 64, 32, 16],
            dropout_rate=0.2
        ).to(self.device)
        
        print(f"\n🧠 Model Architecture:")
        print(self.model)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"📊 Total parameters: {total_params:,}")
        print(f"🎯 Trainable parameters: {trainable_params:,}")
        
    def train_model(self, train_loader, test_loader, epochs=100, learning_rate=0.001):
        """Train the neural network"""
        print("\n" + "="*80)
        print("🚀 MODEL TRAINING")
        print("="*80)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
        
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []
        
        best_test_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validation phase
            self.model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    outputs = self.model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    
                    test_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    test_total += batch_y.size(0)
                    test_correct += (predicted == batch_y).sum().item()
            
            # Calculate averages
            avg_train_loss = train_loss / len(train_loader)
            avg_test_loss = test_loss / len(test_loader)
            train_accuracy = 100 * train_correct / train_total
            test_accuracy = 100 * test_correct / test_total
            
            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
            
            # Learning rate scheduling
            scheduler.step(avg_test_loss)
            
            # Early stopping
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_cancer_model.pth')
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}]")
                print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
                print(f"  Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")
                print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
                print("-" * 50)
            
            # Early stopping
            if patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_cancer_model.pth'))
        
        # Plot training history
        self.plot_training_history(train_losses, test_losses, train_accuracies, test_accuracies)
        
        return train_losses, test_losses, train_accuracies, test_accuracies
    
    def plot_training_history(self, train_losses, test_losses, train_accuracies, test_accuracies):
        """Plot training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(train_losses) + 1)
        
        # Loss plot
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
        ax1.set_title('📉 Model Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, test_accuracies, 'r-', label='Test Accuracy', linewidth=2)
        ax2.set_title('📈 Model Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Loss difference
        loss_diff = np.array(train_losses) - np.array(test_losses)
        ax3.plot(epochs, loss_diff, 'g-', linewidth=2)
        ax3.set_title('🔄 Overfitting Monitor (Train-Test Loss)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss Difference')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.grid(True, alpha=0.3)
        
        # Learning curve
        ax4.plot(epochs, train_accuracies, 'b-', alpha=0.7, label='Training')
        ax4.plot(epochs, test_accuracies, 'r-', alpha=0.7, label='Test')
        ax4.fill_between(epochs, train_accuracies, alpha=0.3, color='blue')
        ax4.fill_between(epochs, test_accuracies, alpha=0.3, color='red')
        ax4.set_title('📊 Learning Curves', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Training history saved as 'training_history.png'")
    
    def evaluate_model(self, test_loader):
        """Evaluate the model and create visualizations"""
        print("\n" + "="*80)
        print("📊 MODEL EVALUATION")
        print("="*80)
        
        self.model.eval()
        y_true = []
        y_pred = []
        y_prob = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                outputs = self.model(batch_X).squeeze()
                predicted = (outputs > 0.5).float()
                
                y_true.extend(batch_y.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_prob.extend(outputs.cpu().numpy())
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)
        
        # Classification report
        print("📋 Classification Report:")
        target_names = ['Benign', 'Malignant']
        print(classification_report(y_true, y_pred, target_names=target_names))
        
        # Create evaluation visualizations
        self.plot_evaluation_results(y_true, y_pred, y_prob, target_names)
        
        return y_true, y_pred, y_prob
    
    def plot_evaluation_results(self, y_true, y_pred, y_prob, target_names):
        """Create evaluation plots"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names, ax=ax1)
        ax1.set_title('🎯 Confusion Matrix', fontsize=14, fontweight='bold')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        ax2.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('📈 ROC Curve', fontsize=14, fontweight='bold')
        ax2.legend(loc="lower right")
        ax2.grid(True, alpha=0.3)
        
        # Prediction Distribution
        benign_probs = y_prob[y_true == 0]
        malignant_probs = y_prob[y_true == 1]
        
        ax3.hist(benign_probs, bins=30, alpha=0.7, label='Benign', color='#4ECDC4', density=True)
        ax3.hist(malignant_probs, bins=30, alpha=0.7, label='Malignant', color='#FF6B6B', density=True)
        ax3.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
        ax3.set_xlabel('Prediction Probability')
        ax3.set_ylabel('Density')
        ax3.set_title('📊 Prediction Distribution', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Performance Metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        values = [accuracy, precision, recall, f1, roc_auc]
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
        
        bars = ax4.bar(metrics, values, color=colors, alpha=0.8)
        ax4.set_ylim([0, 1])
        ax4.set_title('🏆 Performance Metrics', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Evaluation results saved as 'model_evaluation.png'")
        
        # Print summary
        print(f"\n🎯 MODEL PERFORMANCE SUMMARY:")
        print(f"• Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"• Precision: {precision:.3f}")
        print(f"• Recall: {recall:.3f}")
        print(f"• F1-Score: {f1:.3f}")
        print(f"• AUC-ROC: {roc_auc:.3f}")

def main():
    """Main function to run the cancer prediction project"""
    print("🔬" + "="*79)
    print("🔬 BREAST CANCER PREDICTION WITH DEEP LEARNING")
    print("🔬" + "="*79)
    print("🔬 Using PyTorch, Seaborn, Matplotlib, and NumPy")
    print("🔬" + "="*79)
    
    # Data Analysis Phase
    print("\n🔍 PHASE 1: DATA ANALYSIS AND VISUALIZATION")
    analyzer = CancerDataAnalyzer('Cancer_Data.csv')
    analyzer.basic_info()
    analyzer.create_visualizations()
    
    # Model Training Phase
    print("\n🤖 PHASE 2: DEEP LEARNING MODEL TRAINING")
    predictor = CancerPredictor('Cancer_Data.csv')
    
    # Preprocess data
    X_train, X_test, y_train, y_test = predictor.load_and_preprocess_data()
    
    # Create data loaders
    train_loader, test_loader = predictor.create_data_loaders(
        X_train, X_test, y_train, y_test, batch_size=32
    )
    
    # Build and train model
    predictor.build_model(input_size=X_train.shape[1])
    
    # Train the model
    history = predictor.train_model(
        train_loader, test_loader, 
        epochs=150, learning_rate=0.001
    )
    
    # Evaluate the model
    y_true, y_pred, y_prob = predictor.evaluate_model(test_loader)
    
    print("\n" + "="*80)
    print("✅ PROJECT COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("📁 Generated Files:")
    print("  • cancer_data_analysis.png - Data visualization")
    print("  • training_history.png - Training progress")
    print("  • model_evaluation.png - Model performance")
    print("  • best_cancer_model.pth - Trained model weights")
    print("\n🎉 Thank you for using the Cancer Prediction System!")

if __name__ == "__main__":
    main() 