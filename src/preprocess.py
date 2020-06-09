import subprocess
import argparse
from pathlib import Path

def get_pssm(fasta_path, db_path, out_path, thread):
    cmd = ['psiblast', '-query', fasta_path, '-db', db_path, '-out_ascii_pssm', out_path,
            '-num_iterations', '2', '-num_threads', thread, '-save_pssm_after_last_round']
    subprocess.run(cmd)


def predict_local_structure(fasta_path, out_path, thread):
    cmd = ['run_SCRATCH-1D_predictors.sh', fasta_path, out_path, thread]
    subprocess.run(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta_path', '-f', type=str, required=True, help='fasta file path')
    parser.add_argument('--out_dir', '-o', type=str, default='../data/profile', help='output directory')
    parser.add_argument('--db_path', '-d', type=str, required=True, help='uniref90 database path')
    parser.add_argument('--num_thread', '-n', type=str, default='1', help='number of thread')
    args = parser.parse_args()

    out_dir = (Path(args.out_dir)/Path(args.fasta_path).stem).with_suffix('')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir/Path(args.fasta_path).stem
    out_pssm_path = out_path.with_suffix('.pssm')
    out_local_structure_path = out_path.with_suffix('')

    if not out_pssm_path.exists():
        get_pssm(args.fasta_path, args.db_path, out_pssm_path, args.num_thread)
    if not out_local_structure_path.with_suffix('.ss').exists() and not out_local_structure_path.with_suffix('.acc20').exists():
        predict_local_structure(args.fasta_path, out_local_structure_path, args.num_thread)

