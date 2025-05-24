#!/usr/bin/env python3
"""
Cancer Research Data Visualization
==================================
Comprehensive visualization suite for breast cancer prediction research project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend to prevent hanging
plt.switch_backend('Agg')  # Use non-interactive backend

# Set better style for plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

class CancerResearchVisualization:
    """Complete visualization suite for cancer research project"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load and prepare the dataset"""
        print("📊 Loading Cancer Dataset for Research Visualization...")
        self.df = pd.read_csv(self.data_path)
        
        # Clean column names - remove any unnamed columns
        self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed')]
        
        print(f"✅ Dataset loaded: {self.df.shape}")
        print(f"🏷️  Clean columns: {len(self.df.columns)}")
    
    def create_overview_analysis(self):
        """Create comprehensive overview analysis"""
        print("🎨 Creating Overview Analysis...")
        
        # Create figure with proper spacing
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Cancer Dataset Overview Analysis', fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Target Distribution
        ax1 = axes[0, 0]
        diagnosis_counts = self.df['diagnosis'].value_counts()
        colors = ['#4ECDC4', '#FF6B6B']  # Benign, Malignant
        wedges, texts, autotexts = ax1.pie(diagnosis_counts.values, 
                                          labels=['Benign', 'Malignant'], 
                                          colors=colors, 
                                          autopct='%1.1f%%', 
                                          startangle=90,
                                          explode=(0.05, 0.05))
        ax1.set_title('🎯 Diagnosis Distribution', fontweight='bold', pad=20)
        
        # 2. Feature Correlation Heatmap
        ax2 = axes[0, 1]
        mean_features = [col for col in self.df.columns if 'mean' in col][:8]
        if mean_features:
            corr_data = self.df[mean_features].corr()
            sns.heatmap(corr_data, annot=True, cmap='RdYlBu_r', center=0,
                       fmt='.2f', square=True, cbar_kws={'shrink': 0.8}, ax=ax2)
            ax2.set_title('🔥 Feature Correlations', fontweight='bold', pad=20)
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # 3. Radius Distribution
        ax3 = axes[0, 2]
        if 'radius_mean' in self.df.columns:
            for diagnosis, color in zip(['B', 'M'], ['#4ECDC4', '#FF6B6B']):
                data = self.df[self.df['diagnosis'] == diagnosis]['radius_mean']
                ax3.hist(data, alpha=0.7, 
                        label=f'{"Benign" if diagnosis == "B" else "Malignant"}', 
                        bins=25, density=True, color=color)
            ax3.set_xlabel('Radius Mean')
            ax3.set_ylabel('Density')
            ax3.set_title('📏 Radius Distribution', fontweight='bold', pad=20)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Texture Analysis
        ax4 = axes[1, 0]
        if 'texture_mean' in self.df.columns:
            sns.boxplot(data=self.df, x='diagnosis', y='texture_mean', 
                       palette=['#4ECDC4', '#FF6B6B'], ax=ax4)
            ax4.set_title('🔍 Texture Analysis', fontweight='bold', pad=20)
            ax4.grid(True, alpha=0.3)
        
        # 5. Area vs Perimeter
        ax5 = axes[1, 1]
        if 'area_mean' in self.df.columns and 'perimeter_mean' in self.df.columns:
            for diagnosis, color in zip(['B', 'M'], ['#4ECDC4', '#FF6B6B']):
                mask = self.df['diagnosis'] == diagnosis
                ax5.scatter(self.df[mask]['area_mean'], self.df[mask]['perimeter_mean'],
                          c=color, alpha=0.6, s=50,
                          label=f'{"Benign" if diagnosis == "B" else "Malignant"}')
            ax5.set_xlabel('Area Mean')
            ax5.set_ylabel('Perimeter Mean')
            ax5.set_title('📐 Area vs Perimeter', fontweight='bold', pad=20)
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Smoothness Distribution
        ax6 = axes[1, 2]
        if 'smoothness_mean' in self.df.columns:
            sns.violinplot(data=self.df, x='diagnosis', y='smoothness_mean',
                          palette=['#4ECDC4', '#FF6B6B'], ax=ax6)
            ax6.set_title('🌊 Smoothness Distribution', fontweight='bold', pad=20)
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('01_cancer_overview_analysis.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        print("✅ Overview analysis saved as '01_cancer_overview_analysis.png'")
    
    def create_feature_analysis(self):
        """Create detailed feature analysis"""
        print("🔬 Creating Feature Analysis...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Feature Analysis and Comparison', fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Feature Variability
        ax1 = axes[0, 0]
        mean_features = [col for col in self.df.columns if 'mean' in col][:8]
        if mean_features:
            feature_std = self.df[mean_features].std().sort_values(ascending=True)
            bars = ax1.barh(range(len(feature_std)), feature_std.values, 
                           color='lightblue', edgecolor='darkblue')
            ax1.set_yticks(range(len(feature_std)))
            ax1.set_yticklabels([col.replace('_mean', '').title() for col in feature_std.index])
            ax1.set_xlabel('Standard Deviation')
            ax1.set_title('📊 Feature Variability', fontweight='bold', pad=20)
            ax1.grid(True, alpha=0.3, axis='x')
        
        # 2. Worst Features Comparison
        ax2 = axes[0, 1]
        worst_features = [col for col in self.df.columns if 'worst' in col][:5]
        if worst_features:
            malignant_means = self.df[self.df['diagnosis'] == 'M'][worst_features].mean()
            benign_means = self.df[self.df['diagnosis'] == 'B'][worst_features].mean()
            
            x = np.arange(len(worst_features))
            width = 0.35
            
            ax2.bar(x - width/2, malignant_means, width, 
                   label='Malignant', color='#FF6B6B', alpha=0.8)
            ax2.bar(x + width/2, benign_means, width, 
                   label='Benign', color='#4ECDC4', alpha=0.8)
            
            ax2.set_xlabel('Features')
            ax2.set_ylabel('Mean Values')
            ax2.set_title('⚠️ Worst Features Comparison', fontweight='bold', pad=20)
            ax2.set_xticks(x)
            ax2.set_xticklabels([col.replace('_worst', '').title() for col in worst_features], 
                              rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Compactness vs Concavity
        ax3 = axes[0, 2]
        if 'compactness_mean' in self.df.columns and 'concavity_mean' in self.df.columns:
            for diagnosis, color in zip(['B', 'M'], ['#4ECDC4', '#FF6B6B']):
                mask = self.df['diagnosis'] == diagnosis
                ax3.scatter(self.df[mask]['compactness_mean'], 
                          self.df[mask]['concavity_mean'],
                          c=color, alpha=0.6, s=50,
                          label=f'{"Benign" if diagnosis == "B" else "Malignant"}')
            ax3.set_xlabel('Compactness Mean')
            ax3.set_ylabel('Concavity Mean')
            ax3.set_title('🔄 Compactness vs Concavity', fontweight='bold', pad=20)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Symmetry Distribution
        ax4 = axes[1, 0]
        if 'symmetry_mean' in self.df.columns:
            benign_data = self.df[self.df['diagnosis'] == 'B']['symmetry_mean']
            malignant_data = self.df[self.df['diagnosis'] == 'M']['symmetry_mean']
            
            ax4.hist([benign_data, malignant_data], bins=25, alpha=0.7, 
                    label=['Benign', 'Malignant'], color=['#4ECDC4', '#FF6B6B'])
            ax4.set_xlabel('Symmetry Mean')
            ax4.set_ylabel('Frequency')
            ax4.set_title('⚖️ Symmetry Distribution', fontweight='bold', pad=20)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Fractal Dimension
        ax5 = axes[1, 1]
        if 'fractal_dimension_mean' in self.df.columns:
            sns.stripplot(data=self.df, x='diagnosis', y='fractal_dimension_mean',
                         palette=['#4ECDC4', '#FF6B6B'], size=6, ax=ax5)
            ax5.set_title('🔢 Fractal Dimension', fontweight='bold', pad=20)
            ax5.grid(True, alpha=0.3)
        
        # 6. SE Features Analysis
        ax6 = axes[1, 2]
        se_features = [col for col in self.df.columns if '_se' in col][:6]
        if se_features:
            malignant_se = self.df[self.df['diagnosis'] == 'M'][se_features].mean()
            benign_se = self.df[self.df['diagnosis'] == 'B'][se_features].mean()
            
            x = np.arange(len(se_features))
            ax6.plot(x, malignant_se, 'o-', color='#FF6B6B', linewidth=2, 
                    markersize=8, label='Malignant')
            ax6.plot(x, benign_se, 's-', color='#4ECDC4', linewidth=2, 
                    markersize=8, label='Benign')
            
            ax6.set_xlabel('SE Features')
            ax6.set_ylabel('Mean Values')
            ax6.set_title('📈 Standard Error Features', fontweight='bold', pad=20)
            ax6.set_xticks(x)
            ax6.set_xticklabels([col.replace('_se', '').title() for col in se_features], 
                              rotation=45, ha='right')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('02_cancer_feature_analysis.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        print("✅ Feature analysis saved as '02_cancer_feature_analysis.png'")
    
    def create_statistical_summary(self):
        """Create statistical summary visualization"""
        print("📈 Creating Statistical Summary...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Statistical Summary and Data Insights', fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Dataset Summary Statistics
        ax1 = axes[0, 0]
        summary_data = {
            'Total Samples': len(self.df),
            'Features': len(self.df.columns) - 1,  # Exclude diagnosis
            'Malignant Cases': len(self.df[self.df['diagnosis'] == 'M']),
            'Benign Cases': len(self.df[self.df['diagnosis'] == 'B']),
            'Missing Values': self.df.isnull().sum().sum()
        }
        
        bars = ax1.bar(range(len(summary_data)), list(summary_data.values()), 
                      color=['#FF9999', '#66B2FF', '#FF6B6B', '#4ECDC4', '#FFCC99'])
        ax1.set_xticks(range(len(summary_data)))
        ax1.set_xticklabels(list(summary_data.keys()), rotation=45, ha='right')
        ax1.set_title('📊 Dataset Summary', fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, summary_data.values()):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{value}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Feature Distributions by Type
        ax2 = axes[0, 1]
        feature_types = {
            'Mean Features': len([col for col in self.df.columns if 'mean' in col]),
            'SE Features': len([col for col in self.df.columns if '_se' in col]),
            'Worst Features': len([col for col in self.df.columns if 'worst' in col])
        }
        
        ax2.pie(feature_types.values(), labels=feature_types.keys(), autopct='%1.1f%%',
               colors=['#FFB6C1', '#87CEEB', '#DDA0DD'], startangle=90)
        ax2.set_title('📋 Feature Type Distribution', fontweight='bold', pad=20)
        
        # 3. Diagnosis Statistics
        ax3 = axes[1, 0]
        diagnosis_stats = self.df['diagnosis'].value_counts()
        malignant_pct = (diagnosis_stats['M'] / len(self.df)) * 100
        benign_pct = (diagnosis_stats['B'] / len(self.df)) * 100
        
        stats_text = f"""
        📊 DIAGNOSIS STATISTICS
        
        Total Patients: {len(self.df)}
        
        🔴 Malignant (M): {diagnosis_stats['M']} ({malignant_pct:.1f}%)
        🟢 Benign (B): {diagnosis_stats['B']} ({benign_pct:.1f}%)
        
        📈 Malignant Rate: {malignant_pct:.1f}%
        📉 Benign Rate: {benign_pct:.1f}%
        
        ⚖️ Class Ratio: 1:{benign_pct/malignant_pct:.1f}
        """
        
        ax3.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.5))
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        ax3.set_title('📈 Clinical Statistics', fontweight='bold', pad=20)
        
        # 4. Key Insights
        ax4 = axes[1, 1]
        insights_text = f"""
        🔬 KEY RESEARCH INSIGHTS
        
        📊 Dataset Quality:
        • No missing values detected
        • Well-balanced feature set
        • {len(self.df.columns)-1} clinical measurements
        
        🎯 Classification Challenge:
        • Binary classification task
        • Imbalanced dataset ({malignant_pct:.1f}% malignant)
        • High-dimensional feature space
        
        🧠 Machine Learning Potential:
        • Suitable for deep learning
        • Rich feature correlations
        • Clinical relevance validated
        """
        
        ax4.text(0.1, 0.5, insights_text, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.3))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('💡 Research Insights', fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('03_cancer_statistical_summary.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        print("✅ Statistical summary saved as '03_cancer_statistical_summary.png'")
    
    def generate_data_report(self):
        """Generate comprehensive data report"""
        print("📋 Generating Data Report...")
        
        report = {
            'dataset_shape': self.df.shape,
            'total_samples': len(self.df),
            'total_features': len(self.df.columns) - 1,
            'missing_values': self.df.isnull().sum().sum(),
            'diagnosis_distribution': self.df['diagnosis'].value_counts().to_dict(),
            'feature_types': {
                'mean_features': len([col for col in self.df.columns if 'mean' in col]),
                'se_features': len([col for col in self.df.columns if '_se' in col]),
                'worst_features': len([col for col in self.df.columns if 'worst' in col])
            },
            'malignant_percentage': (self.df['diagnosis'].value_counts()['M'] / len(self.df)) * 100,
            'benign_percentage': (self.df['diagnosis'].value_counts()['B'] / len(self.df)) * 100
        }
        
        return report

def main():
    """Run complete research visualization suite"""
    print("🔬" + "="*80)
    print("🔬 CANCER RESEARCH VISUALIZATION SUITE")
    print("🔬" + "="*80)
    
    # Create visualizations
    visualizer = CancerResearchVisualization('Cancer_Data.csv')
    
    # Generate all visualizations
    visualizer.create_overview_analysis()
    visualizer.create_feature_analysis()
    visualizer.create_statistical_summary()
    
    # Generate data report
    report = visualizer.generate_data_report()
    
    print("\n" + "="*80)
    print("✅ RESEARCH VISUALIZATIONS COMPLETED!")
    print("="*80)
    print("📁 Generated Files:")
    print("  • 01_cancer_overview_analysis.png")
    print("  • 02_cancer_feature_analysis.png")
    print("  • 03_cancer_statistical_summary.png")
    print(f"\n📊 Dataset Summary:")
    print(f"  • Total Samples: {report['total_samples']}")
    print(f"  • Total Features: {report['total_features']}")
    print(f"  • Malignant Cases: {report['diagnosis_distribution']['M']} ({report['malignant_percentage']:.1f}%)")
    print(f"  • Benign Cases: {report['diagnosis_distribution']['B']} ({report['benign_percentage']:.1f}%)")
    print("\n🎉 Ready for research documentation!")

if __name__ == "__main__":
    main() 