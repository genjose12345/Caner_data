#!/usr/bin/env python3
"""
Improved Cancer Data Visualization
=================================
Enhanced version with better layout and readability
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set better style for plots
plt.style.use('default')  # Use default instead of seaborn-v0_8 which might not be available
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

class ImprovedCancerVisualization:
    """Improved visualization class with better layout"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load and prepare the dataset"""
        print("📊 Loading Cancer Dataset for Improved Visualization...")
        self.df = pd.read_csv(self.data_path)
        
        # Clean column names - remove any unnamed columns
        self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed')]
        
        print(f"✅ Dataset loaded: {self.df.shape}")
        print(f"🏷️  Clean columns: {len(self.df.columns)}")
    
    def create_overview_plots(self):
        """Create overview plots with better layout"""
        print("🎨 Creating Overview Visualizations...")
        
        # Create figure with better spacing
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Target Distribution (Enhanced Pie Chart)
        ax1 = fig.add_subplot(gs[0, 0])
        diagnosis_counts = self.df['diagnosis'].value_counts()
        colors = ['#FF6B6B', '#4ECDC4']
        wedges, texts, autotexts = ax1.pie(diagnosis_counts.values, 
                                          labels=['Benign', 'Malignant'], 
                                          colors=colors, 
                                          autopct='%1.1f%%', 
                                          startangle=90,
                                          explode=(0.05, 0.05),
                                          shadow=True)
        ax1.set_title('🎯 Cancer Diagnosis Distribution', fontsize=16, fontweight='bold', pad=20)
        
        # Make text larger and more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')
        
        # 2. Enhanced Correlation Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        # Select mean features for correlation
        mean_features = [col for col in self.df.columns if 'mean' in col][:8]  # Top 8 for readability
        if mean_features:
            corr_data = self.df[mean_features].corr()
            im = sns.heatmap(corr_data, annot=True, cmap='RdYlBu_r', center=0,
                           fmt='.2f', square=True, cbar_kws={'shrink': 0.8}, ax=ax2)
            ax2.set_title('🔥 Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=10)
            plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=10)
        
        # 3. Enhanced Distribution Plot
        ax3 = fig.add_subplot(gs[0, 2])
        if 'radius_mean' in self.df.columns:
            for diagnosis, color in zip(['B', 'M'], ['#4ECDC4', '#FF6B6B']):
                data = self.df[self.df['diagnosis'] == diagnosis]['radius_mean']
                ax3.hist(data, alpha=0.7, 
                        label=f'{"Benign" if diagnosis == "B" else "Malignant"}', 
                        bins=25, density=True, color=color, edgecolor='black', linewidth=0.5)
            ax3.set_xlabel('Radius Mean', fontsize=12)
            ax3.set_ylabel('Density', fontsize=12)
            ax3.set_title('📏 Radius Mean Distribution', fontsize=16, fontweight='bold', pad=20)
            ax3.legend(fontsize=12)
            ax3.grid(True, alpha=0.3)
        
        # 4. Enhanced Box Plot
        ax4 = fig.add_subplot(gs[1, 0])
        if 'texture_mean' in self.df.columns:
            box_plot = sns.boxplot(data=self.df, x='diagnosis', y='texture_mean', 
                                 palette=['#4ECDC4', '#FF6B6B'], ax=ax4)
            ax4.set_title('🔍 Texture Mean by Diagnosis', fontsize=16, fontweight='bold', pad=20)
            ax4.set_xlabel('Diagnosis', fontsize=12)
            ax4.set_ylabel('Texture Mean', fontsize=12)
            ax4.grid(True, alpha=0.3)
        
        # 5. Enhanced Scatter Plot
        ax5 = fig.add_subplot(gs[1, 1])
        if 'area_mean' in self.df.columns and 'perimeter_mean' in self.df.columns:
            for diagnosis, color in zip(['B', 'M'], ['#4ECDC4', '#FF6B6B']):
                mask = self.df['diagnosis'] == diagnosis
                ax5.scatter(self.df[mask]['area_mean'], self.df[mask]['perimeter_mean'],
                          c=color, alpha=0.7, s=60, edgecolors='black', linewidth=0.5,
                          label=f'{"Benign" if diagnosis == "B" else "Malignant"}')
            ax5.set_xlabel('Area Mean', fontsize=12)
            ax5.set_ylabel('Perimeter Mean', fontsize=12)
            ax5.set_title('📐 Area vs Perimeter Mean', fontsize=16, fontweight='bold', pad=20)
            ax5.legend(fontsize=12)
            ax5.grid(True, alpha=0.3)
        
        # 6. Enhanced Violin Plot
        ax6 = fig.add_subplot(gs[1, 2])
        if 'smoothness_mean' in self.df.columns:
            violin_plot = sns.violinplot(data=self.df, x='diagnosis', y='smoothness_mean',
                                       palette=['#4ECDC4', '#FF6B6B'], ax=ax6)
            ax6.set_title('🌊 Smoothness Distribution', fontsize=16, fontweight='bold', pad=20)
            ax6.set_xlabel('Diagnosis', fontsize=12)
            ax6.set_ylabel('Smoothness Mean', fontsize=12)
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cancer_data_overview_improved.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        print("✅ Overview plots saved as 'cancer_data_overview_improved.png'")
    
    def create_detailed_analysis(self):
        """Create detailed analysis plots"""
        print("🔬 Creating Detailed Analysis...")
        
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Feature Importance Analysis
        ax1 = fig.add_subplot(gs[0, 0])
        mean_features = [col for col in self.df.columns if 'mean' in col][:8]
        if mean_features:
            feature_std = self.df[mean_features].std().sort_values(ascending=True)
            bars = ax1.barh(range(len(feature_std)), feature_std.values, 
                           color='lightblue', edgecolor='darkblue', linewidth=1)
            ax1.set_yticks(range(len(feature_std)))
            ax1.set_yticklabels([col.replace('_mean', '').title() for col in feature_std.index], 
                              fontsize=11)
            ax1.set_xlabel('Standard Deviation', fontsize=12)
            ax1.set_title('📊 Feature Variability', fontsize=16, fontweight='bold', pad=20)
            ax1.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax1.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                        f'{width:.1f}', ha='left', va='center', fontsize=10)
        
        # 2. Worst Features Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        worst_features = [col for col in self.df.columns if 'worst' in col][:5]
        if worst_features:
            malignant_means = self.df[self.df['diagnosis'] == 'M'][worst_features].mean()
            benign_means = self.df[self.df['diagnosis'] == 'B'][worst_features].mean()
            
            x = np.arange(len(worst_features))
            width = 0.35
            
            bars1 = ax2.bar(x - width/2, malignant_means, width, 
                           label='Malignant', color='#FF6B6B', alpha=0.8, 
                           edgecolor='darkred', linewidth=1)
            bars2 = ax2.bar(x + width/2, benign_means, width, 
                           label='Benign', color='#4ECDC4', alpha=0.8,
                           edgecolor='darkcyan', linewidth=1)
            
            ax2.set_xlabel('Features', fontsize=12)
            ax2.set_ylabel('Mean Values', fontsize=12)
            ax2.set_title('⚠️ Worst Features Comparison', fontsize=16, fontweight='bold', pad=20)
            ax2.set_xticks(x)
            ax2.set_xticklabels([col.replace('_worst', '').title() for col in worst_features], 
                              rotation=45, ha='right', fontsize=10)
            ax2.legend(fontsize=12)
            ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Advanced Scatter Analysis
        ax3 = fig.add_subplot(gs[0, 2])
        if 'compactness_mean' in self.df.columns and 'concavity_mean' in self.df.columns:
            for diagnosis, color in zip(['B', 'M'], ['#4ECDC4', '#FF6B6B']):
                mask = self.df['diagnosis'] == diagnosis
                ax3.scatter(self.df[mask]['compactness_mean'], 
                          self.df[mask]['concavity_mean'],
                          c=color, alpha=0.7, s=60, edgecolors='black', linewidth=0.5,
                          label=f'{"Benign" if diagnosis == "B" else "Malignant"}')
            ax3.set_xlabel('Compactness Mean', fontsize=12)
            ax3.set_ylabel('Concavity Mean', fontsize=12)
            ax3.set_title('🔄 Compactness vs Concavity', fontsize=16, fontweight='bold', pad=20)
            ax3.legend(fontsize=12)
            ax3.grid(True, alpha=0.3)
        
        # 4. Enhanced Histogram
        ax4 = fig.add_subplot(gs[1, 0])
        if 'symmetry_mean' in self.df.columns:
            benign_data = self.df[self.df['diagnosis'] == 'B']['symmetry_mean']
            malignant_data = self.df[self.df['diagnosis'] == 'M']['symmetry_mean']
            
            ax4.hist([benign_data, malignant_data], bins=25, alpha=0.7, 
                    label=['Benign', 'Malignant'], color=['#4ECDC4', '#FF6B6B'],
                    edgecolor='black', linewidth=0.5)
            ax4.set_xlabel('Symmetry Mean', fontsize=12)
            ax4.set_ylabel('Frequency', fontsize=12)
            ax4.set_title('⚖️ Symmetry Distribution', fontsize=16, fontweight='bold', pad=20)
            ax4.legend(fontsize=12)
            ax4.grid(True, alpha=0.3)
        
        # 5. Strip Plot
        ax5 = fig.add_subplot(gs[1, 1])
        if 'fractal_dimension_mean' in self.df.columns:
            strip_plot = sns.stripplot(data=self.df, x='diagnosis', y='fractal_dimension_mean',
                                     palette=['#4ECDC4', '#FF6B6B'], size=8, alpha=0.7, ax=ax5)
            ax5.set_title('🔢 Fractal Dimension Distribution', fontsize=16, fontweight='bold', pad=20)
            ax5.set_xlabel('Diagnosis', fontsize=12)
            ax5.set_ylabel('Fractal Dimension Mean', fontsize=12)
            ax5.grid(True, alpha=0.3)
        
        # 6. SE Features Analysis
        ax6 = fig.add_subplot(gs[1, 2])
        se_features = [col for col in self.df.columns if '_se' in col][:6]
        if se_features:
            malignant_se = self.df[self.df['diagnosis'] == 'M'][se_features].mean()
            benign_se = self.df[self.df['diagnosis'] == 'B'][se_features].mean()
            
            x = np.arange(len(se_features))
            ax6.plot(x, malignant_se, 'o-', color='#FF6B6B', linewidth=3, 
                    markersize=10, label='Malignant', markeredgecolor='darkred', markeredgewidth=2)
            ax6.plot(x, benign_se, 's-', color='#4ECDC4', linewidth=3, 
                    markersize=10, label='Benign', markeredgecolor='darkcyan', markeredgewidth=2)
            
            ax6.set_xlabel('SE Features', fontsize=12)
            ax6.set_ylabel('Mean Values', fontsize=12)
            ax6.set_title('📈 Standard Error Features', fontsize=16, fontweight='bold', pad=20)
            ax6.set_xticks(x)
            ax6.set_xticklabels([col.replace('_se', '').title() for col in se_features], 
                              rotation=45, ha='right', fontsize=10)
            ax6.legend(fontsize=12)
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cancer_detailed_analysis_improved.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.show()
        print("✅ Detailed analysis saved as 'cancer_detailed_analysis_improved.png'")
    
    def create_summary_stats(self):
        """Create summary statistics"""
        print("📈 Dataset Summary Statistics:")
        print("="*60)
        print(f"🔍 Total Samples: {len(self.df)}")
        print(f"📊 Total Features: {len(self.df.columns)}")
        print(f"🎯 Missing Values: {self.df.isnull().sum().sum()}")
        
        if 'diagnosis' in self.df.columns:
            diagnosis_counts = self.df['diagnosis'].value_counts()
            print(f"\n📋 Diagnosis Distribution:")
            print(f"  • Malignant (M): {diagnosis_counts.get('M', 0)} ({diagnosis_counts.get('M', 0)/len(self.df)*100:.1f}%)")
            print(f"  • Benign (B): {diagnosis_counts.get('B', 0)} ({diagnosis_counts.get('B', 0)/len(self.df)*100:.1f}%)")

def main():
    """Run improved visualization"""
    print("🎨" + "="*60)
    print("🎨 IMPROVED CANCER DATA VISUALIZATION")
    print("🎨" + "="*60)
    
    # Create improved visualizations
    visualizer = ImprovedCancerVisualization('Cancer_Data.csv')
    
    # Display summary
    visualizer.create_summary_stats()
    
    # Create visualizations
    visualizer.create_overview_plots()
    visualizer.create_detailed_analysis()
    
    print("\n" + "="*60)
    print("✅ IMPROVED VISUALIZATIONS COMPLETED!")
    print("="*60)
    print("📁 Generated Files:")
    print("  • cancer_data_overview_improved.png")
    print("  • cancer_detailed_analysis_improved.png")
    print("\n🎉 Much better layout and readability!")

if __name__ == "__main__":
    main() 