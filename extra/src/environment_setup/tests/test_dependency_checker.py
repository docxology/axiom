"""
Tests for Dependency Checker

Comprehensive tests for the DependencyChecker class.
"""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

# Add the src directory to the path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from environment_setup.dependency_checker import DependencyChecker, DependencyError


class TestDependencyChecker(unittest.TestCase):
    """Tests for DependencyChecker class."""

    def setUp(self):
        """Set up test fixtures."""
        self.checker = DependencyChecker(verbose=False)

    def test_initialization(self):
        """Test DependencyChecker initialization."""
        checker = DependencyChecker(verbose=True)
        self.assertTrue(checker.verbose)

        checker_no_verbose = DependencyChecker(verbose=False)
        self.assertFalse(checker_no_verbose.verbose)

    @patch('environment_setup.dependency_checker.importlib.import_module')
    def test_is_package_installed_true(self, mock_import):
        """Test checking installed package."""
        mock_import.return_value = MagicMock()
        result = self.checker.is_package_installed('numpy')
        self.assertTrue(result)
        mock_import.assert_called_with('numpy')

    @patch('environment_setup.dependency_checker.importlib.import_module')
    def test_is_package_installed_false(self, mock_import):
        """Test checking non-installed package."""
        mock_import.side_effect = ImportError("No module named 'nonexistent'")
        result = self.checker.is_package_installed('nonexistent')
        self.assertFalse(result)

    @patch('environment_setup.dependency_checker.importlib.import_module')
    def test_get_package_version(self, mock_import):
        """Test getting package version."""
        mock_module = MagicMock()
        mock_module.__version__ = '1.21.0'
        mock_import.return_value = mock_module

        version = self.checker.get_package_version('numpy')
        self.assertEqual(version, '1.21.0')

    @patch('environment_setup.dependency_checker.importlib.import_module')
    def test_get_package_version_no_version(self, mock_import):
        """Test getting package version when not available."""
        mock_module = MagicMock()
        del mock_module.__version__  # No version attribute
        mock_import.return_value = mock_module

        version = self.checker.get_package_version('numpy')
        self.assertIsNone(version)

    @patch('environment_setup.dependency_checker.importlib.import_module')
    def test_get_package_version_not_installed(self, mock_import):
        """Test getting package version for non-installed package."""
        mock_import.side_effect = ImportError("No module named 'nonexistent'")
        version = self.checker.get_package_version('nonexistent')
        self.assertIsNone(version)

    @patch.object(DependencyChecker, 'is_package_installed')
    @patch.object(DependencyChecker, 'get_package_version')
    def test_check_requirements(self, mock_get_version, mock_is_installed):
        """Test checking multiple requirements."""
        requirements = ['numpy>=1.21.0', 'torch>=2.0.0', 'nonexistent>=1.0.0']

        def mock_is_installed_side_effect(package):
            if package.startswith('nonexistent'):
                return False
            return True

        mock_is_installed.side_effect = mock_is_installed_side_effect
        mock_get_version.return_value = '1.21.0'

        installed, missing = self.checker.check_requirements(requirements)

        self.assertEqual(len(installed), 2)
        self.assertEqual(len(missing), 1)
        self.assertIn('nonexistent>=1.0.0', missing)
        self.assertIn('numpy>=1.21.0', installed)
        self.assertIn('torch>=2.0.0', installed)

    @patch('environment_setup.dependency_checker.subprocess.run')
    def test_install_package_success(self, mock_run):
        """Test successful package installation."""
        mock_run.return_value = MagicMock()

        result = self.checker.install_package('numpy>=1.21.0')
        self.assertTrue(result)

        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        self.assertIn('numpy>=1.21.0', args[0])

    @patch('environment_setup.dependency_checker.subprocess.run')
    def test_install_package_failure(self, mock_run):
        """Test failed package installation."""
        mock_run.side_effect = subprocess.CalledProcessError(1, 'pip', stderr='Installation failed')

        result = self.checker.install_package('nonexistent-package')
        self.assertFalse(result)

    @patch('environment_setup.dependency_checker.subprocess.run')
    def test_install_package_timeout(self, mock_run):
        """Test package installation timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired('pip install', 300)

        result = self.checker.install_package('slow-package')
        self.assertFalse(result)

    @patch('environment_setup.dependency_checker.subprocess.run')
    def test_install_package_unexpected_error(self, mock_run):
        """Test package installation with unexpected error."""
        mock_run.side_effect = Exception('Unexpected error')

        result = self.checker.install_package('problematic-package')
        self.assertFalse(result)

    @patch.object(DependencyChecker, 'install_package')
    def test_install_packages(self, mock_install):
        """Test installing multiple packages."""
        packages = ['numpy>=1.21.0', 'torch>=2.0.0', 'pandas']

        def mock_install_side_effect(package, upgrade=False):
            if package == 'pandas':
                return False
            return True

        mock_install.side_effect = mock_install_side_effect

        successful, failed = self.checker.install_packages(packages)

        self.assertEqual(len(successful), 2)
        self.assertEqual(len(failed), 1)
        self.assertIn('pandas', failed)
        self.assertIn('numpy>=1.21.0', successful)
        self.assertIn('torch>=2.0.0', successful)

    @patch('environment_setup.dependency_checker.subprocess.run')
    def test_install_requirements_file_success(self, mock_run):
        """Test successful installation from requirements file."""
        mock_run.return_value = MagicMock()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('numpy>=1.21.0\n')
            f.write('torch>=2.0.0\n')
            temp_path = Path(f.name)

        try:
            result = self.checker.install_requirements_file(temp_path)
            self.assertTrue(result)

            mock_run.assert_called_once()
            args, kwargs = mock_run.call_args
            self.assertIn('-r', args[0])
            self.assertEqual(args[0][-1], str(temp_path))
        finally:
            temp_path.unlink()

    @patch('environment_setup.dependency_checker.subprocess.run')
    def test_install_requirements_file_not_found(self, mock_run):
        """Test installing from non-existent requirements file."""
        with self.assertRaises(DependencyError):
            self.checker.install_requirements_file('/nonexistent/requirements.txt')

    @patch('environment_setup.dependency_checker.subprocess.run')
    def test_install_requirements_file_failure(self, mock_run):
        """Test failed installation from requirements file."""
        mock_run.side_effect = subprocess.CalledProcessError(1, 'pip', stderr='Installation failed')

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('numpy>=1.21.0\n')
            temp_path = Path(f.name)

        try:
            result = self.checker.install_requirements_file(temp_path)
            self.assertFalse(result)
        finally:
            temp_path.unlink()

    def test_create_requirements_file(self):
        """Test creating requirements file."""
        packages = ['numpy>=1.21.0', 'torch>=2.0.0', 'pandas']

        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_path = Path(f.name)

        try:
            self.checker.create_requirements_file(packages, temp_path)

            self.assertTrue(temp_path.exists())

            with open(temp_path, 'r') as f:
                content = f.read().strip()

            lines = content.split('\n')
            self.assertEqual(len(lines), 3)
            for package in packages:
                self.assertIn(package, lines)
        finally:
            temp_path.unlink()

    @patch('environment_setup.dependency_checker.sys')
    def test_verify_python_version(self, mock_sys):
        """Test Python version verification."""
        # Test matching version
        mock_sys.version_info = type('version_info', (), {'major': 3, 'minor': 11})()
        result = self.checker.verify_python_version('3.11')
        self.assertTrue(result)

        # Test higher version required
        result = self.checker.verify_python_version('3.9')
        self.assertTrue(result)

        # Test lower version
        result = self.checker.verify_python_version('3.12')
        self.assertFalse(result)

        # Test invalid version format
        result = self.checker.verify_python_version('invalid')
        self.assertFalse(result)

    @patch('environment_setup.dependency_checker.subprocess.run')
    def test_check_system_dependencies(self, mock_run):
        """Test checking system dependencies."""
        def mock_run_side_effect(cmd, *args, **kwargs):
            if cmd[0] in ['git', 'curl', 'wget', 'make']:
                return MagicMock()
            elif cmd[0] == 'gcc':
                raise subprocess.CalledProcessError(1, 'gcc')
            else:
                raise FileNotFoundError()

        mock_run.side_effect = mock_run_side_effect

        deps = self.checker.check_system_dependencies()

        expected_deps = ['git', 'curl', 'wget', 'make', 'gcc']
        for dep in expected_deps:
            self.assertIn(dep, deps)

        self.assertTrue(deps['git'])
        self.assertTrue(deps['curl'])
        self.assertTrue(deps['wget'])
        self.assertTrue(deps['make'])
        self.assertFalse(deps['gcc'])

    @patch.object(DependencyChecker, 'is_package_installed')
    @patch.object(DependencyChecker, 'get_package_version')
    @patch('environment_setup.dependency_checker.sys')
    def test_get_environment_summary(self, mock_sys, mock_get_version, mock_is_installed):
        """Test getting environment summary."""
        mock_sys.version_info = type('version_info', (), {'major': 3, 'minor': 11, 'micro': 5})()
        mock_sys.executable = '/usr/bin/python3'

        def mock_is_installed_side_effect(package):
            return package in ['torch', 'numpy', 'pandas']

        mock_is_installed.side_effect = mock_is_installed_side_effect
        mock_get_version.return_value = '1.21.0'

        summary = self.checker.get_environment_summary()

        self.assertIn('python_version', summary)
        self.assertIn('python_executable', summary)
        self.assertIn('system_dependencies', summary)
        self.assertIn('python_packages', summary)

        self.assertEqual(summary['python_version'], '3.11.5')
        self.assertEqual(summary['python_executable'], '/usr/bin/python3')

        packages = summary['python_packages']
        self.assertIn('torch', packages)
        self.assertIn('numpy', packages)
        self.assertIn('pandas', packages)
        self.assertIn('matplotlib', packages)  # Should be in summary even if not installed

        self.assertTrue(packages['torch']['installed'])
        self.assertEqual(packages['torch']['version'], '1.21.0')
        self.assertFalse(packages['matplotlib']['installed'])


if __name__ == '__main__':
    unittest.main()
