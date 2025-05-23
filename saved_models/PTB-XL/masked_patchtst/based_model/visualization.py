import pandas as pd
import matplotlib.pyplot as plt

# Set these manually:
save_path = './'  # or the directory where you want to save the plots
save_finetuned_model = 'PTB-XL_patchtst_finetuned_cw512_tw96_patch12_stride12_epochs-finetune20_model1'

df = pd.read_csv('PTB-XL_patchtst_finetuned_cw512_tw96_patch12_stride12_epochs-finetune20_model1_per_class_report.csv', index_col=0)
class_cols = [col for col in df.columns if col.isdigit()]
metrics = df.index.tolist()

for metric in metrics:
    plt.figure(figsize=(16,4))
    plt.bar(class_cols, df.loc[metric, class_cols].astype(float))
    plt.title(f'Per-class {metric}')
    plt.xlabel('Class')
    plt.ylabel(metric)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{save_path}{save_finetuned_model}_per_class_{metric}.png")
    plt.close()