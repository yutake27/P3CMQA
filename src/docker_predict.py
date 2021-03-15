import argparse
import os
import subprocess
from pathlib import Path
from typing import List


class Docker:
    def __init__(self, image_name: str, gpu_id: int, container_name: str):
        self.image_name = image_name
        self.gpu_id = gpu_id
        self.user_info = str(os.getuid()) + ':' + str(os.getgid())
        self.container_id = None
        self.container_name = container_name

    def _run_cmd_err_handle(self, cmd) -> subprocess.CompletedProcess:
        cp = subprocess.run(cmd, capture_output=True, text=True)
        if cp.stderr:
            print(cp.stderr)
            quit()
        return cp

    def run_detached(self, volume_dir_list: List[str]) -> None:
        # basic command
        cmd = ['docker', 'run', '-dit', '--rm', '-u', self.user_info]
        # gpu
        if self.gpu_id >= 0:
            cmd.append('--gpus=' + str(self.gpu_id))
        # volume
        for volume_dir in volume_dir_list:
            cmd.extend(['-v', volume_dir])
        # name
        cmd.extend(['--name', self.container_name, self.image_name])
        print(' '.join(cmd))
        cp = self._run_cmd_err_handle(cmd)
        container_id = cp.stdout.strip()
        self.container_id = container_id
        print('Container ID:', container_id)

    def exec(self, workspace: str, cmd_list: List[str]) -> None:
        assert self.container_id
        cmd = ['docker', 'exec', '-it', '-u', self.user_info, '-w', workspace, self.container_name] + cmd_list
        print(' '.join(cmd))
        subprocess.run(cmd)

    def stop(self) -> None:
        assert self.container_id
        cmd = ['docker', 'stop', self.container_id]
        subprocess.run(cmd)


def Rel2AbsPath(RelPath: str) -> str:
    return str(Path(RelPath).resolve())


def getParentAbsPath(RelPath: str) -> str:
    return str(Path(RelPath).resolve().parent)


class DockerPredict:
    def __init__(self, image_name: str, gpu_id: int, pdb_file_rpath: str, pdb_dir_rpath: str,
                 fasta_rpath: str, profile_dir_rpath: str, output_dir_rpath: str):
        """Predict with Docker

        Args:
            image_name (str): the name of the image. For example, yutake27/p3cmqa:cuda11.0-cudnn8
            gpu_id (int): GPU ID. if you do not use gpu, specify -1.
            pdb_file_rpath (str): relative/absolute path of single pdb file.
            pdb_dir_rpath (str): relative/absolute path of pdb directory.
            fasta_rpath (str): relative/absolute path of fasta file.
            profile_dir_rpath (str): relative/absolute path of profile directory.
            output_dir_rpath (str): relative/absolute path of output directory.

        """
        self.image_name: str = image_name
        self.gpu_id: int = gpu_id
        self.container_name: str = 'p3cmqa'
        self.target_name: str = Path(fasta_rpath).stem
        self.pdb_file_path: str = Rel2AbsPath(pdb_file_rpath) if pdb_file_rpath else None
        self.pdb_dir_path: str = Rel2AbsPath(pdb_dir_rpath) if pdb_dir_rpath else getParentAbsPath(pdb_file_rpath)
        self.fasta_dir_path: str = getParentAbsPath(fasta_rpath)
        profile_dir_rpath = profile_dir_rpath if profile_dir_rpath else '../data/profile/' + self.target_name
        output_dir_rpath = output_dir_rpath if output_dir_rpath else '../data/score/' + self.target_name
        Path(output_dir_rpath).mkdir(parents=True, exist_ok=True)
        self.profile_dir_path: str = Rel2AbsPath(profile_dir_rpath)
        self.output_dir_path: str = Rel2AbsPath(output_dir_rpath)
        self.volume_dir_list: List[str] = self.get_volume_dir_list()

    def get_volume_dir_list(self) -> List[str]:
        repository_dir_volume = str(Path.cwd().parent) + ':/home'
        pdb_dir_volume = str(self.pdb_dir_path) + ':/home/data/pdb/' + self.target_name
        fasta_dir_volume = str(self.fasta_dir_path) + ':/home/data/fasta/'
        profile_dir_volume = str(self.profile_dir_path) + ':/home/data/profile/' + self.target_name
        output_dir_volume = str(self.output_dir_path) + ':/home/data/score/' + self.target_name
        return [repository_dir_volume, pdb_dir_volume, fasta_dir_volume, profile_dir_volume, output_dir_volume]

    def get_predict_cmd_str(self) -> (List[str], str):
        workspace = '/home/src'
        predict_cmd_list = ['python', 'predict.py']
        if self.pdb_file_path:
            model_file_name = Path(self.pdb_file_path).name
            predict_cmd_list.extend(['-i', '../data/pdb/' + self.target_name + '/' + model_file_name])
        else:
            predict_cmd_list.extend(['-d', '../data/pdb/' + self.target_name])
        predict_cmd_list.extend(['-f', '../data/fasta/' + self.target_name + '.fasta', '-g', str(self.gpu_id)])
        return workspace, predict_cmd_list

    def predict(self) -> None:
        docker = Docker(self.image_name, self.gpu_id, self.container_name)
        print('#### docker run ####')
        docker.run_detached(self.volume_dir_list)
        print('#### docker exec ####')
        workspace, predict_cmd_list = self.get_predict_cmd_str()
        docker.exec(workspace, predict_cmd_list)
        print('#### docker stop ####')
        docker.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run P3CMQA with Docker')
    parser.add_argument('docker_image', type=str,
                        help='The name of pulled docker image. For example, "yutake27/p3cmqa:cuda11.0-cudnn8"')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU). If you use a GPU, please specify the GPU ID.')
    parser.add_argument('--input_file_path', '-i',
                        help='Input single pdb path.\n\
                        Use this option if you want to make predictions for a single pdb file.')
    parser.add_argument('--input_dir_path', '-d',
                        help='Input directory path.\n\
                        Use this option if you want to make predictions for multiple model structures in the same directory.')
    parser.add_argument('--fasta_path', '-f', required=True, help='Reference FASTA Sequence path.')
    parser.add_argument('--profile_dir', '-p', type=str,
                        help='Path of input profile directory. Default) ../data/profile/target_name')
    parser.add_argument('--output_dir', '-o', type=str,
                        help='Path of Output directory. Default) ../data/score/target_name')
    # parser.add_argument('--save_res', '-s', action='store_true', help='save score of each residue')
    args = parser.parse_args()

    dp = DockerPredict(args.docker_image, args.gpu, args.input_file_path, args.input_dir_path,
                       args.fasta_path, args.profile_dir, args.output_dir)
    dp.predict()
