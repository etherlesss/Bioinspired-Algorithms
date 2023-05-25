import subprocess
import pkg_resources

class d_installer:
    def __init__(self, d_path: str) -> None:
        # Leer lista de dependencias desde la lista
        with open(d_path, 'r') as file:
            self.dependencies = [line.strip() for line in file.readlines()]

        self.install_dependencies()

    # Instalador de dependencias
    def install_dependencies(self):
        for dependency in self.dependencies:
            if self.is_dependency_installed(dependency):
                print(f"{dependency} is already installed.")
            else:
                try:
                    subprocess.check_call(["pip", "install", dependency])
                    print(f"Successfully installed {dependency}")
                except subprocess.CalledProcessError as e:
                    print(f"Failed to install {dependency}. Error: {e}")

    # Verificador de dependencias
    @staticmethod
    def is_dependency_installed(dependency):
        try:
            pkg_resources.get_distribution(dependency)
            return True
        except pkg_resources.DistributionNotFound:
            return False
