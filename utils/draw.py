import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse


mpl.rcParams['font.family']   = 'serif'
mpl.rcParams['font.serif']    = ['Georgia']
mpl.rcParams['font.size']     = 20
mpl.rcParams['axes.titlesize']= 20
mpl.rcParams['axes.labelsize']= 18
mpl.rcParams['xtick.labelsize']=16
mpl.rcParams['ytick.labelsize']=16
# no legend, so no need to set legend.fontsize

def plot_two_loss_curves(
    csv_file1,
    csv_file2,
    title="Loss Comparison on Qwen3-8B",
    dataset1_name="Dataset1",
    dataset2_name="Dataset2"
):
    # Read CSV files
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)

    # Check columns
    for df, path in ((df1, csv_file1), (df2, csv_file2)):
        if 'Step' not in df.columns or 'Loss' not in df.columns:
            raise ValueError(f"Missing 'Step' or 'Loss' columns in {path}")

    # Create figure
    plt.figure(figsize=(12, 8))

    # Plot two lines with softer colors
    plt.plot(df1['Step'], df1['Loss'],
             color='#1f77b4', linewidth=2.5)  # steel blue
    plt.plot(df2['Step'], df2['Loss'],
             color='#2ca02c', linewidth=2.5)  # medium sea green

    # Title and labels
    plt.title(title, fontweight='bold')
    plt.xlabel('Steps', fontweight='bold')
    plt.ylabel('Loss',  fontweight='bold')

    # Grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Layout
    plt.tight_layout(pad=3.0)

    # Save
    plt.savefig('loss_comparison_qwen38b.svg', format='svg')
    plt.savefig('loss_comparison.png', dpi=300)

    # Display
    plt.show()

    print("Saved: loss_comparison.svg, loss_comparison.png")


def main():
    parser = argparse.ArgumentParser(description='Plot comparison of two training loss curves')
    parser.add_argument('csv_file1', help='Path to the first CSV file')
    parser.add_argument('csv_file2', help='Path to the second CSV file')
    parser.add_argument('--title', default='Training Loss Comparison', help='Title for the plot')
    parser.add_argument('--dataset1-name', default='Original Dataset', help='Name for the first dataset')
    parser.add_argument('--dataset2-name', default='Revised Dataset', help='Name for the second dataset')
    
    args = parser.parse_args()
    
    plot_two_loss_curves(
        args.csv_file1,
        args.csv_file2,
        title=args.title,
        dataset1_name=args.dataset1_name,
        dataset2_name=args.dataset2_name
    )


if __name__ == "__main__":
    main()