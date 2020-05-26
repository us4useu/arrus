import unittest
import tempfile
import os
from pathlib import Path
from install import UpdateEnvVariablesStage

class UpdateEnvVariablesStageTest(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()

    def create_file(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").close()

    def create_tmp_files(self, files_to_create):
        for file in files_to_create:
            path = os.path.join(self.tmp_dir.name, file)
            self.create_file(path)

    def create_tmp_dirs(self, dirs_to_create):
        result = []
        for dir in dirs_to_create:
            path = os.path.join(self.tmp_dir.name, dir)
            os.makedirs(path)
            result.append(path)
        return result

    def test_update_paths_list_first_path(self):
        files = [r"arrus\lib64\Us4OEM.dll"]
        dirs = [r"arrus\lib64", r"windows", r"windows\system32", r"office"]
        tmp_dirs = self.create_tmp_dirs(dirs)
        self.create_tmp_files(files)

        install_path = os.path.join(self.tmp_dir.name, r"arrus")

        UpdateEnvVariablesStage.update_paths_list(
            paths_list=tmp_dirs,
            install_path=install_path
        )
        self.assertEqual(tmp_dirs[0], os.path.join(install_path, "lib64"))

    def test_update_paths_list_second_path(self):
        files = [r"arrus\lib64\Us4OEM.dll"]
        dirs = [r"windows", r"arrus\lib64", r"windows\system32", r"office"]
        tmp_dirs = self.create_tmp_dirs(dirs)
        self.create_tmp_files(files)

        install_path = os.path.join(self.tmp_dir.name, r"arrus")

        UpdateEnvVariablesStage.update_paths_list(
            paths_list=tmp_dirs,
            install_path=install_path
        )
        self.assertEqual(tmp_dirs[1], os.path.join(install_path, "lib64"))

    def test_update_paths_last_path(self):
        files = [r"arrus\lib64\Us4OEM.dll"]
        dirs = [r"windows", r"windows\system32", r"office", r"arrus\lib64"]
        tmp_dirs = self.create_tmp_dirs(dirs)
        self.create_tmp_files(files)

        install_path = os.path.join(self.tmp_dir.name, r"arrus")

        UpdateEnvVariablesStage.update_paths_list(
            paths_list=tmp_dirs,
            install_path=install_path
        )
        self.assertEqual(tmp_dirs[-1], os.path.join(install_path, "lib64"))

    def test_update_path_different_current_path(self):
        files = [r"arrus-old\lib64\Us4OEM.dll"]
        dirs = [r"arrus-old\lib64", r"windows", r"windows\system32", r"office"]
        tmp_dirs = self.create_tmp_dirs(dirs)
        self.create_tmp_files(files)

        install_path = os.path.join(self.tmp_dir.name, r"arrus")

        UpdateEnvVariablesStage.update_paths_list(
            paths_list=tmp_dirs,
            install_path=install_path
        )
        self.assertEqual(len(tmp_dirs), 5)
        self.assertEqual(tmp_dirs[0], Path(os.path.join(install_path, "lib64")))

    def test_update_path_different_current_path_2(self):
        files = [r"arrus-old\lib64\Us4OEM.dll"]
        dirs = [r"windows", r"arrus-old\lib64", r"windows\system32", r"office"]
        tmp_dirs = self.create_tmp_dirs(dirs)
        self.create_tmp_files(files)

        install_path = os.path.join(self.tmp_dir.name, r"arrus")

        UpdateEnvVariablesStage.update_paths_list(
            paths_list=tmp_dirs,
            install_path=install_path
        )
        self.assertEqual(len(tmp_dirs), 5)
        self.assertEqual(tmp_dirs[1], Path(os.path.join(install_path, "lib64")))

    def test_update_path_different_current_path_last(self):
        files = [r"arrus-old\lib64\Us4OEM.dll"]
        dirs = [r"windows", r"windows\system32", r"office", r"arrus-old\lib64"]
        tmp_dirs = self.create_tmp_dirs(dirs)
        self.create_tmp_files(files)

        install_path = os.path.join(self.tmp_dir.name, r"arrus")

        UpdateEnvVariablesStage.update_paths_list(
            paths_list=tmp_dirs,
            install_path=install_path
        )
        self.assertEqual(len(tmp_dirs), 5)
        print(tmp_dirs)
        self.assertEqual(tmp_dirs[3], Path(os.path.join(install_path, "lib64")))

    def test_update_path_no_current_path(self):
        dirs = [r"windows", r"windows\system32", r"office"]
        tmp_dirs = self.create_tmp_dirs(dirs)

        install_path = os.path.join(self.tmp_dir.name, r"arrus")

        UpdateEnvVariablesStage.update_paths_list(
            paths_list=tmp_dirs,
            install_path=install_path
        )
        self.assertEqual(len(tmp_dirs), 4)
        print(tmp_dirs)
        self.assertEqual(tmp_dirs[3], Path(os.path.join(install_path, "lib64")))

if __name__ == "__main__":
    unittest.main()
