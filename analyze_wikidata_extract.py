import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
def load_groups_from_csv(csv_path, group_col='id', data_col='name', min_large=20, min_small=1):
    df = pd.read_csv(csv_path)
    groups = df.groupby(group_col)[data_col].apply(list)
    large_groups = [group for group in groups if len(group) >= min_large]
    small_groups = [group for group in groups if min_small <= len(group) < min_large]
    return large_groups, small_groups


# -----------------------------------------------------------------------------
def choose_cutoff(sizes: pd.Series, target_train_groups=100_000, target_val_groups=10_000, target_test_groups=10_000, min_size=4, buffer_ratio=0.10):

    sizes = sizes[sizes >= min_size]
    size_freq = sizes.value_counts().sort_index()

    # descending cumulative sum (index: size, value=#groups >= size)
    desc = size_freq[::-1].cumsum()[::-1]

    total_target = int((target_train_groups + target_val_groups + target_test_groups) * (1.0 + buffer_ratio))

    # smallest cutoff who's cumulative >= total_target
    viable = desc[desc >= total_target]
    if viable.empty:
        # nothing meets the target, fall back to min
        return int(min_size)
    return int(viable.index.min())


# -----------------------------------------------------------------------------
def stratified_group_split(sizes: pd.Series,
                           cutoff: int,
                           train=0.8, val=0.1, test=0.1,
                           bins=(10, 20, 50, 100, 200, np.inf),
                           random_state=42):
    """
    Returns three arrays of group IDs: train_ids, val_ids, test_ids.
    Stratifies by size bucket to keep distribution stable across splits.
    """
    assert abs(train + val + test - 1.0) < 1e-9
    keep = sizes[sizes >= cutoff].copy()
    # assign bucket labels
    # edges like: (cutoff-1, 10], (10,20], (20,50], ... (200, inf]
    edges = [cutoff - 1] + list(bins)
    labels = [f"{int(edges[i])+1}-{('∞' if np.isinf(edges[i+1]) else int(edges[i+1]))}" for i in range(len(edges) - 1)]
    buckets = pd.cut(keep, bins=edges, labels=labels, right=True, include_lowest=True)

    rng = np.random.RandomState(random_state)
    train_ids, val_ids, test_ids = [], [], []

    for label in buckets.cat.categories:
        idx = keep.index[buckets == label]
        if len(idx) == 0:
            continue
        # shuffle
        idx = np.array(idx)
        rng.shuffle(idx)

        n = len(idx)
        n_train = int(round(n * train))
        n_val = int(round(n * val))
        # ensure all assigned
        # n_test = n - n_train - n_val

        train_ids.extend(idx[:n_train])
        val_ids.extend(idx[n_train:n_train + n_val])
        test_ids.extend(idx[n_train + n_val:])

    return np.array(train_ids), np.array(val_ids), np.array(test_ids)


# -----------------------------------------------------------------------------
def plot_histogram(sizes, path, min_small=2, bin_size=10, cap=200):

    # lower end cutoff
    sizes = sizes[sizes >= min_small]

    # cap large values
    sizes_capped = np.where(sizes > cap, cap, sizes)

    # create bins
    bins = list(range(0, cap + bin_size, bin_size))
    labels = [f"{b + 1}-{b + bin_size}" for b in bins[:-1]]
    labels[-1] = f"{cap - bin_size + 1}+"

    # bin the sizes
    sizes_binned = pd.cut(sizes_capped, bins=bins, labels=labels, right=True, include_lowest=True)

    # build a freqency table per bin
    counts = sizes_binned.value_counts().reindex(labels, fill_value=0)

    # plot
    counts.plot(kind='bar')
    plt.title("Name Group Size Distributions")
    plt.xlabel("Group size (# names)")
    plt.ylabel("# groups")
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'name_group_distributions.png'), bbox_inches='tight', dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
def plot_group_sizes_binned(sizes, path, min_small=0, bin_size=10, cap=200):

    # 1. groups to keep
    sizes = sizes[sizes >= min_small]

    # sanity check
    total_groups = len(sizes)
    print(f"Total groups after cutoff (min size={min_small}): {total_groups}")

    # 2. bin groups: 0-9, 10-19, ... 100+
    edges = [0] + list(range(10, cap + 1, bin_size)) + [np.inf]
    labels = [f"{start}-{start+bin_size-1}" for start in range(1, cap, bin_size)] + [f"{cap}+"]

    # 3. assign bins
    binned = pd.cut(sizes, bins=edges, labels=labels, right=True, include_lowest=True)

    # 4. count groups per bin
    counts = binned.value_counts().reindex(labels, fill_value=0)

    # sanity check
    print(f"Sum of bins: {counts.sum()}")
    assert counts.sum() == total_groups, "Bin counts should add up to total groups."

    # 5. create the chart
    ax = counts.plot(kind="bar")
    ax.set_xlabel("Group size range (#names)")
    ax.set_ylabel("# groups")
    ax.set_title(f"Distribution of name group sizes (>= {min_small}, {bin_size}-wide bins, {cap}+ capped)")
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'name_group_distributions.png'), bbox_inches='tight', dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
def cutoff_explorer(sizes, path, min_small=2, candidate_cutoffs=(5, 10, 20, 50, 100), max_plot_cutoff=200, suggest_max_groups=None, suggest_max_names=None, show_plot=True):

    # apply lower cutoff
    sizes = sizes[sizes >= min_small]
    total_groups = len(sizes)
    total_names = int(sizes.sum())

    # frequency of sizes
    size_freq = sizes.value_counts().sort_index()  # index: size, value: #groups
    names_at_size = size_freq.index.to_series() * size_freq  # number of names contributing

    # build base table by size
    base = pd.DataFrame({
        "size": size_freq.index,
        "num_groups_at_size": size_freq.values,
        "num_names_at_size": names_at_size.values
    })

    # cumulative >= size
    base_desc = base.sort_values("size", ascending=False).copy()
    base_desc["cum_groups_ge"] = base_desc["num_groups_at_size"].cumsum()
    base_desc["cum_names_ge"] = base_desc["num_names_at_size"].cumsum()
    summary = base_desc.sort_values("size").copy()

    # percentages
    summary["cum_groups_ge_pct"] = summary["cum_groups_ge"] / total_groups * 100.0
    summary["cum_names_ge_pct"] = summary["cum_names_ge"] / total_names * 100.0

    # print quick summary for candidate cutoffs
    cand = pd.DataFrame({"size": sorted(set(candidate_cutoffs))}).merge(summary, on="size", how="left")
    print("\n--- Candidate cutoff summary (keeping groups with size >= cutoff) ---")
    cols = ["size", "cum_groups_ge", "cum_groups_ge_pct", "cum_names_ge", "cum_names_ge_pct"]
    print(cand[cols].to_string(index=False, formatters={
        "cum_groups_ge_pct": "{:.2f}%".format,
        "cum_names_ge_pct": "{:.2f}%".format,
    }))

    # possible suggestions
    suggestion = None
    if suggest_max_groups is not None or suggest_max_names is not None:
        # find the smallest cutoff
        ok = summary.copy()
        if suggest_max_groups is not None:
            ok = ok[ok["cum_groups_ge"] <= suggest_max_groups]
        if suggest_max_names is not None:
            ok = ok[ok["cum_names_ge"] <= suggest_max_names]
        if not ok.empty:
            suggestion = int(ok["size"].min())
            print(f"\nSuggested cutoff: {suggestion} (where max groups = {suggest_max_groups} and max_names = {suggest_max_names})\n")
        else:
            print("\nNo cutoff meets the provided caps; consider raising themor the cutoff.")

    # plot cumulative vs cutoff (>= cutoff)
    if show_plot:
        # Use only up to max_plot_cutoff for x-axis readability; last point shows ">= max_plot_cutoff"
        plot_df = summary[summary["size"] <= max_plot_cutoff].copy()
        if summary["size"].max() > max_plot_cutoff:
            # Append a synthetic "cap+" point representing everything >= max_plot_cutoff
            cap_row = summary[summary["size"] >= max_plot_cutoff].iloc[0]
            plot_df = pd.concat([plot_df, cap_row.to_frame().T], ignore_index=True)
            plot_df.loc[plot_df.index[-1], "size"] = max_plot_cutoff
            # plot_df.iloc[-1, plot_df.columns.get_loc("size")] = max_plot_cutoff  # collapse to cap

        fig, ax = plt.subplots()
        ax.plot(plot_df["size"], plot_df["cum_groups_ge_pct"], label="% groups ≥ size")
        ax.plot(plot_df["size"], plot_df["cum_names_ge_pct"], label="% names ≥ size")
        ax.set_xlabel("Cutoff (keep groups with size ≥ cutoff)")
        ax.set_ylabel("Cumulative percentage")
        ax.set_title("Cutoff Explorer: cumulative % of groups and names (≥ cutoff)")
        ax.legend()
        ax.grid(True, which="both", axis="both")  # optional; remove if you prefer
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'cumulativeVcutoff.png'), bbox_inches='tight', dpi=150)
        plt.close()


# -----------------------------------------------------------------------------
def estimate_triplets(sizes: pd.Series, group_ids, k_neg_per_pair=1):
    """
    For groups in group_ids, estimate the number of possible triplets if you:
      - pick (anchor, positive) as ordered pairs within a group: s*(s-1)
      - pair each with k_neg_per_pair negatives from other groups (rough estimate)
    """
    sel = sizes.loc[group_ids]
    # number of ordered anchor-positive pairs across all groups
    pos_pairs = (sel * (sel - 1)).sum()
    # negatives available = all examples outside the current group; we approximate with k_neg_per_pair
    return int(pos_pairs * k_neg_per_pair)


# -----------------------------------------------------------------------------
def load_groups_from_csv_and_plot(csv_path, analysis_path, group_col='id', data_col='name', min_large=20, min_small=2):
    df = pd.read_csv(csv_path)

    # 1. group sizes
    sizes = df.groupby(group_col)[data_col].size()

    # 2. indices for large/small groups
    # large_ids = sizes.index[sizes >= min_large]
    # small_ids = sizes.index[(sizes >= min_small) & (sizes < min_large)]

    # 3. build lists
    # grouped_lists = df.groupby(group_col)[data_col].apply(list)
    # large_groups = grouped_lists.loc[large_ids].tolist()
    # small_groups = grouped_lists.loc[small_ids].tolist()

    # 2+3. get group sizes
    large_count = (sizes >= min_large).sum()
    small_count = ((sizes >= min_small) & (sizes < min_large)).sum()

    # 4. create histogram
    # plot_histogram(sizes.value_counts().sort_index(), analysis_path)
    # plot_group_sizes_binned(sizes.value_counts().sort_index(), analysis_path)
    cutoff_explorer(sizes, analysis_path, candidate_cutoffs=(5, 10, 20, 30, 40, 50, 75, 100, 150, 200), suggest_max_groups=100_000)
    cutoff = choose_cutoff(sizes, min_size=min_small)
    print(f"Chosen cutoff (>= size): {cutoff}")
    # Example split:
    train_ids, val_ids, test_ids = stratified_group_split(sizes, cutoff=cutoff, train=0.8, val=0.1, test=0.1, bins=(20, 50, 100, 200, np.inf), random_state=42)
    # print(len(train_ids), len(val_ids), len(test_ids))
    triplets_train = estimate_triplets(sizes, train_ids, k_neg_per_pair=2)
    triplets_val = estimate_triplets(sizes, val_ids, k_neg_per_pair=2)
    triplets_test = estimate_triplets(sizes, test_ids, k_neg_per_pair=2)

    print("Triplet capacity (approx):")
    print(f"\ttrain:\t{triplets_train:>15,}")
    print(f"\tval:\t{triplets_val:>15,}")
    print(f"\ttest:\t{triplets_test:>15,}")

    # return large_groups, small_groups
    return large_count, small_count, len(sizes)


# =============================================================================
# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="analyze_wikidata_extract", description="Performs a categorical frequency count on the specified column."
    )

    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file.')
    parser.add_argument('--out_path', type=str, required=True, help='Base path for output.')
    parser.add_argument('--group_col', type=str, required=False, default='id', help='Column header in CSV file that denotes the group. Default \'id\'')
    parser.add_argument('--data_col', type=str, required=False, default='name', help='Column header in CSV file that denotes the data. Default \'name\'')
    parser.add_argument('--min_large', type=int, required=False, default=20, help='Defines the minium size of a \'large\' group. Default 20')
    parser.add_argument('--min_small', type=int, required=False, default=1, help='Defines the minium size of a \'small\' group. Default 1')
    # parser.add_argument("-o", "--outfile", action="store", required=True)
    # parser.add_argument("infile", type=str, help="File of lines to phrase")
    args = parser.parse_args()

    csv_path = args.csv_path
    base_path, _ = os.path.splitext(csv_path)
    os.makedirs(args.out_path, exist_ok=True)
    analysis_path = os.path.join(args.out_path, "analysis")
    os.makedirs(analysis_path, exist_ok=True)
    print(f"📌 output path = {args.out_path}")
    print(f"📌 analysis path = {analysis_path}")
    # outfile_path = args.outfile
    # line_count = 0
    # print("☺", flush=True, end='\r')
    # df = pd.read_csv(infile_path)
    # category_counts = df[column].value_counts()
    # if (threshold):
    #     categories_to_keep = category_counts[category_counts >= threshold].index
    #     df_trimmed = df[df[column].isin(categories_to_keep)]
    #     df_trimmed.to_csv(base_path + '-trimmed.csv', index=False)
    #     start_unique_count = df[column].nunique()
    #     trimmed_unique_count = df_trimmed[column].nunique()
    #     print("Number of unique " + column + f"s: {start_unique_count} --> {trimmed_unique_count}", flush=True)

    # large_groups, small_groups = load_groups_from_csv(csv_path, group_col=args.group_col, data_col=args.data_col, min_large=args.min_large, min_small=args.min_small)
    # print(f"{len(large_groups):>10} Large groups")
    # print(f"{len(small_groups):>10} Small groups")
    # print(f"{len(large_groups) + len(small_groups):>10} Total groups")
    large_count, small_count, total = load_groups_from_csv_and_plot(csv_path, analysis_path, group_col=args.group_col, data_col=args.data_col, min_large=args.min_large, min_small=args.min_small)
    print(f"{large_count:>10} Large groups")
    print(f"{small_count:>10} Small groups")
    print(f"{total:>10} Total groups")

    # # Using Pandas value_counts() and plot()
    # category_counts.plot(kind='bar')
    # plt.title('Frequency of ' + column + ' (Pandas)')
    # plt.xlabel(column)
    # plt.ylabel('count')
    # plt.savefig(base_path + '-pandas.png')
    # # plt.show()

    # # Using Seaborn countplot()
    # sns.countplot(x=column, data=df)
    # plt.title('Frequency of ' + column + ' (Seaborn)')
    # plt.xlabel(column)
    # plt.ylabel('count')
    # plt.savefig(base_path + '-seaborn.png')
    # # plt.show()
