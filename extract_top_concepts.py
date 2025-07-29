import argparse
import torch
import csv

def main():
    parser = argparse.ArgumentParser(
        description="Extract top-influencing concepts per label from TCAV scores"
    )
    parser.add_argument('--tcav_path',       required=True,
                        help='TCAV scores .pth file (dict concept→{label→score})')
    parser.add_argument('--labels',          type=int, nargs='+', required=True,
                        help='List of label indices to process')
    parser.add_argument('--concepts',        nargs='+', required=True,
                        help='List of all concept names (keys in the TCAV dict)')
    parser.add_argument('--score_threshold', type=float, default=0.7,
                        help='TCAV score threshold for initial filtering')
    parser.add_argument('--min_k',           type=int,   default=3,
                        help='Minimum number of concepts per label')
    parser.add_argument('--max_k',           type=int,   default=10,
                        help='Maximum number of concepts per label')
    parser.add_argument('--output_csv',      required=True,
                        help='Where to save the CSV of top concepts per label')
    args = parser.parse_args()

    tcav_scores = torch.load(args.tcav_path)

    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'concept', 'score', 'rank'])

        for label in args.labels:
            cs = []
            for concept in args.concepts:
                score = tcav_scores.get(concept, {}).get(label, float('nan'))
                cs.append((concept, score))
            cs_sorted = sorted(cs, key=lambda x: (x[1] if x[1]==x[1] else -1), reverse=True)

            selected = [(c,s) for c,s in cs_sorted if s >= args.score_threshold]

            if len(selected) < args.min_k:
                selected = cs_sorted[:args.min_k]
            elif len(selected) > args.max_k:
                selected = selected[:args.max_k]

            for rank, (concept, score) in enumerate(selected, start=1):
                writer.writerow([label, concept, f"{score:.4f}", rank])

    print(f"[INFO] Top concepts per label saved to {args.output_csv}")

if __name__ == '__main__':
    main()
