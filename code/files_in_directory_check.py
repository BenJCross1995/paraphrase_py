#!/usr/bin/env python3

import os
import sys
import argparse

def read_ground_truth(file_path):
    """
    Read the known document list from a text file.
    Each line should represent one document name (without a file extension).
    Empty lines are ignored.
    """
    with open(file_path, 'r') as f:
        return set(line.strip() for line in f if line.strip())

def get_base_names(directory):
    """
    Return a set of base names (file names without extensions) for all files in the given directory.
    """
    return {os.path.splitext(f)[0] for f in os.listdir(directory)}

def check_stage(stage_label, stage_dir, previous_set, ground_truth, exit_on_missing=True):
    """
    Check a pipeline stage by comparing the files in a given directory (stage_dir)
    to both the previous stage and the ground truth.
    
    It prints:
      - The percentage of files present relative to the ground truth.
      - The newly dropped files from the previous stage.
      - The cumulative missing files compared to ground truth.
    
    Parameters:
      stage_label (str): Name of the stage (e.g., "Rephrased", "Parascore", "Top Impostors")
      stage_dir (str): The directory path for this stage.
      previous_set (set): The file set from the previous stage.
      ground_truth (set): The complete ground truth set.
      exit_on_missing (bool): If True, exit if new files are missing.
    
    Returns:
      current_set (set): The file set at the current stage.
    """
    if not os.path.isdir(stage_dir):
        print(f"Error: {stage_label} directory '{stage_dir}' does not exist.")
        sys.exit(1)
    
    # Get files for current stage (filtered to only those in the ground truth)
    current_set = get_base_names(stage_dir) & ground_truth
    missing_new = previous_set - current_set
    cumulative_missing = ground_truth - current_set
    pct = (len(current_set) / len(ground_truth)) * 100 if ground_truth else 0
    
    print(f"\nStage: {stage_label}")
    print(f"Files present: {len(current_set)}/{len(ground_truth)} ({pct:.2f}% remaining)")
    print("Newly dropped (from previous stage):", sorted(missing_new))
    print("Cumulative missing (from ground truth):", sorted(cumulative_missing))
    if missing_new and exit_on_missing:
        print(f"Error: Missing files in the {stage_label} stage. Exiting.")
        sys.exit(1)
    return current_set

def run_pipeline(known_doc_list, rephrased, parascore, top_impostors):
    """
    Run the cascade check for pipeline stages:
      ground truth -> rephrased -> parascore -> top impostors.
    
    Each stage is checked against ground truth (for percentages) and against the previous stage 
    (for newly dropped files). Only the provided stages are computed and printed in the final report.
    
    If not all stages are provided, the script exits on missing files (exit_on_missing=True).
    If all stages are provided, exit_on_missing is set to False so that the full report is printed.
    """
    # Load ground truth
    gt = read_ground_truth(known_doc_list)
    if not gt:
        print("Error: Ground truth file is empty.")
        sys.exit(1)
    print("Ground Truth: {} files".format(len(gt)))
    
    # Determine whether to exit on missing files.
    # If all stages are provided, set exit_on_missing=False to allow full reporting.
    exit_on_missing = not (rephrased and parascore and top_impostors)
    
    # Compute each stage only if provided.
    if rephrased:
        rephrased_set = check_stage("Rephrased", rephrased, gt, gt, exit_on_missing=exit_on_missing)
    else:
        rephrased_set = None

    if parascore:
        if rephrased_set is None:
            print("Error: Cannot check Parascore stage because Rephrased stage is not provided.")
            sys.exit(1)
        parascore_set = check_stage("Parascore", parascore, rephrased_set, gt, exit_on_missing=exit_on_missing)
    else:
        parascore_set = None

    if top_impostors:
        if parascore_set is None:
            print("Error: Cannot check Top Impostors stage because Parascore stage is not provided.")
            sys.exit(1)
        top_impostors_set = check_stage("Top Impostors", top_impostors, parascore_set, gt, exit_on_missing=exit_on_missing)
    else:
        top_impostors_set = None

    # Final report: print only the stages provided
    print("\nFull Report:")
    print("Ground Truth: {} files".format(len(gt)))
    if rephrased:
        print("Rephrased: {} files".format(len(rephrased_set)))
    if parascore:
        print("Parascore: {} files".format(len(parascore_set)))
    if top_impostors:
        print("Top Impostors: {} files".format(len(top_impostors_set)))
    
    if top_impostors:
        overall_missing = sorted(gt - top_impostors_set)
        print("\nOverall missing (from ground truth to Top Impostors):", overall_missing)
    
    return rephrased_set, parascore_set, top_impostors_set

def main():
    parser = argparse.ArgumentParser(
        description="Cascade file check for pipeline stages. Compares files at each stage "
                    "against the ground truth and the previous stage. Only the provided stages "
                    "will be printed in the final report."
    )
    parser.add_argument("--known_doc_list", required=True,
                        help="Path to the ground truth document list (base names without file types)")
    parser.add_argument("--rephrased", help="Path to the rephrased directory")
    parser.add_argument("--parascore", help="Path to the parascore directory")
    parser.add_argument("--top_impostors", help="Path to the top_impostors directory")
    
    args = parser.parse_args()
    run_pipeline(args.known_doc_list, args.rephrased, args.parascore, args.top_impostors)

if __name__ == "__main__":
    main()
