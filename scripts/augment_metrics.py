"""Post-experiment metrics augmentation: Add PPV and NPV to all results."""
import json
import csv
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

def compute_ppv_npv(cm):
    """CM format: [[TN, FP], [FN, TP]]"""
    TN, FP = cm[0]
    FN, TP = cm[1]
    ppv = round(TP / (TP + FP), 4) if (TP + FP) > 0 else 0.0
    npv = round(TN / (TN + FN), 4) if (TN + FN) > 0 else 0.0
    return ppv, npv


def main():
    experiments = [
        'resnet_baseline',
        'clip_zero_shot',
        'clip_linear_probe',
        'clip_finetune',
        'hyperbolic_clip',
        'hyperbolic_zero_shot',
    ]

    results_table = []

    for exp in experiments:
        path = PROJECT / 'experiments' / exp / 'results.json'
        if not path.exists():
            print(f'SKIP: {exp} — results.json not found')
            continue
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        if 'confusion_matrix' not in data:
            print(f'SKIP: {exp} — no confusion_matrix')
            continue
        
        cm = data['confusion_matrix']
        ppv, npv = compute_ppv_npv(cm)
        data['PPV'] = ppv
        data['NPV'] = npv
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        results_table.append({
            'Model': exp.replace('_', ' ').title(),
            'Accuracy': round(data.get('accuracy', 0), 4),
            'Precision': round(data.get('precision', 0), 4),
            'Recall': round(data.get('recall', 0), 4),
            'F1': round(data.get('f1', 0), 4),
            'AUROC': round(data.get('auroc', 0), 4),
            'AUPRC': round(data.get('auprc', 0), 4),
            'PPV': ppv,
            'NPV': npv,
        })
        print(f'UPDATED: {exp}  PPV={ppv}  NPV={npv}')

    # Handle resnet_cross_generator (summary file, compute average from folds)
    cross_gen_path = PROJECT / 'experiments' / 'resnet_cross_generator' / 'summary_results.json'
    if cross_gen_path.exists():
        with open(cross_gen_path, 'r') as f:
            cg_data = json.load(f)
        
        folds = list(cg_data.keys())
        avg = {
            'accuracy': sum(cg_data[f]['accuracy'] for f in folds) / len(folds),
            'precision': sum(cg_data[f]['precision'] for f in folds) / len(folds),
            'recall': sum(cg_data[f]['recall'] for f in folds) / len(folds),
            'f1': sum(cg_data[f]['f1'] for f in folds) / len(folds),
            'auroc': sum(cg_data[f]['auroc'] for f in folds) / len(folds),
            'auprc': sum(cg_data[f]['auprc'] for f in folds) / len(folds),
        }
        # PPV = precision for the positive class
        ppv_cg = round(avg['precision'], 4)
        
        results_table.append({
            'Model': 'Resnet Cross Generator (Avg)',
            'Accuracy': round(avg['accuracy'], 4),
            'Precision': round(avg['precision'], 4),
            'Recall': round(avg['recall'], 4),
            'F1': round(avg['f1'], 4),
            'AUROC': round(avg['auroc'], 4),
            'AUPRC': round(avg['auprc'], 4),
            'PPV': ppv_cg,
            'NPV': 'N/A',
        })
        print(f'INCLUDED (avg): resnet_cross_generator  PPV={ppv_cg}  NPV=N/A (no CM)')

    # Sort by logical experiment order
    order = ['Resnet Baseline', 'Resnet Cross Generator (Avg)', 'Clip Zero Shot',
             'Hyperbolic Zero Shot', 'Clip Linear Probe', 'Clip Finetune', 'Hyperbolic Clip']
    results_table.sort(key=lambda x: order.index(x['Model']) if x['Model'] in order else 99)

    # Generate markdown
    md_lines = [
        '# Final Model Comparison',
        '',
        '| Model | Accuracy | Precision | Recall | F1 | AUROC | AUPRC | PPV | NPV |',
        '|-------|----------|-----------|--------|-----|-------|-------|-----|-----|',
    ]
    for r in results_table:
        npv_str = str(r['NPV']) if r['NPV'] != 'N/A' else 'N/A'
        md_lines.append(
            f"| {r['Model']} | {r['Accuracy']} | {r['Precision']} | {r['Recall']} | "
            f"{r['F1']} | {r['AUROC']} | {r['AUPRC']} | {r['PPV']} | {npv_str} |"
        )
    
    md_lines.extend([
        '',
        '> PPV = Positive Predictive Value = TP / (TP + FP)',
        '>',
        '> NPV = Negative Predictive Value = TN / (TN + FN)',
        '',
        'Generated automatically from experiment results.',
    ])

    md_path = PROJECT / 'docs' / 'final_model_comparison.md'
    md_path.parent.mkdir(exist_ok=True)
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f'SAVED: {md_path}')

    # Generate CSV
    csv_path = PROJECT / 'experiments' / 'final_model_comparison.csv'
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUROC', 'AUPRC', 'PPV', 'NPV']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_table)
    print(f'SAVED: {csv_path}')

    print(f'\n{"="*40}')
    print(f'Experiments processed: {len(experiments) + 1}')
    print(f'PPV added: ✓')
    print(f'NPV added: ✓')
    print(f'Updated files: results.json in each experiment')
    print(f'Comparison table saved: docs/final_model_comparison.md')
    print(f'CSV saved: experiments/final_model_comparison.csv')


if __name__ == '__main__':
    main()
